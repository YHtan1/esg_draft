"""Replaceable NumPy backend. No per-path Python objects or random consumption."""
from dataclasses import dataclass
import time
import numpy as np
from .market import make_grid
from .credit import CorporateBond, corporate_value
from .rates import B, joint_gaussian_kernel
from ._random import RandomSource


@dataclass
class Paths:
    times: np.ndarray
    reporting_indices: np.ndarray
    names: tuple
    state_names: tuple
    gains: np.ndarray
    states: np.ndarray
    discounts: np.ndarray
    fx: np.ndarray
    defaults: np.ndarray
    diagnostics: dict
    values: np.ndarray = None
    discounted_distributions: np.ndarray = None


def simulate(market, paths=1000, horizon=10., internal_steps=12,
             seed=7, scheme="exact", terminal_only=False,
             heston_scheme="full_truncation", store_security_values=True):
    """Exactly six securities under domestic THB Q, with monthly reporting.

    Order: Thai/US equity, Thai/US 10Y government zero, Thai/US 10Y corporate.
    FX is THB per USD. Foreign prices and distributions are converted into THB.
    Dividends are paid at step end as S_end*(exp(q*dt)-1); refine the grid to
    assess this approximation. terminal_only saves endpoints for convergence.
    """
    government_maturities = (10.,)
    corporate_maturity = 10.
    recovery_intervals = 8
    simulation_start = time.perf_counter()
    if (not isinstance(paths,int) or paths < 1 or scheme not in ("exact","euler")
            or heston_scheme not in ("full_truncation","reflection")
):
        raise ValueError("Invalid simulation settings")
    if horizon > 10.:
        raise ValueError("The six securities have a ten-year projection horizon")
    bonds = tuple(CorporateBond("corporate_"+label,ccy,corporate_maturity) for label,ccy in
                  zip(("local","foreign"),(market.domestic,market.foreign)))
    # Curve interpolation knots are not security cash-flow dates. Exact shift
    # integrals already handle a step crossing an interpolation knot.
    cashflows = np.r_[bonds[0].times,bonds[1].times,government_maturities,
                      market.credits[0].knots,market.credits[1].knots]
    times, reporting = make_grid(horizon,internal_steps,12,cashflows)
    if horizon > min(c.knots[-1] for c in market.credits):
        raise ValueError("Simulation exceeds fitted credit horizon")
    saved = np.array([0,len(times)-1]) if terminal_only else reporting
    save_lookup = {int(k):i for i,k in enumerate(saved)}
    names = market.drivers
    d, m = len(names), len(market.rates[0].mean_reversions)
    rate_ids = [names.index(p+"_"+label) for label in ("local","foreign") for p in (("x",) if m==1 else ("x","y"))]
    a = np.array([v for r in market.rates for v in r.mean_reversions])
    sig = np.array([v for r in market.rates for v in r.volatilities])
    r = np.asarray(market.correlation)
    fx_id = names.index("FX")
    force = np.zeros(2*m)
    force[m:] = -sig[m:]*market.fx_vol*r[rate_ids[m:],fx_id]
    gaussian_dim = d+4*m
    source = RandomSource(seed,np.arange(paths),times,gaussian_dim)
    thresholds = -np.log(source.default_uniforms())
    factors, v, z = np.zeros((paths,2*m)), np.tile([e.v0 for e in market.equities],(paths,1)), np.ones((paths,2))
    spots = np.tile([e.spot for e in market.equities],(paths,1))
    fx, discount = np.full(paths,market.fx_spot), np.ones((paths,2))
    hazard, alive = np.zeros((paths,2)), np.ones((paths,2),dtype=bool)
    default_times = np.full((paths,2),np.inf)
    assets = ("equity_local","equity_foreign")+tuple("government_"+label for label in ("local","foreign") for t in government_maturities)+("corporate_local","corporate_foreign")
    state_names = ("equity_local","equity_foreign","v_local","v_foreign")+tuple(names[j] for j in rate_ids)+("credit_local","credit_foreign","FX")
    cumulative = np.zeros((paths,len(assets)))
    gains = np.empty((len(saved),paths,len(assets)))
    security_values = np.empty_like(gains) if store_security_values else None
    distributions = np.empty_like(gains) if store_security_values else None
    states = np.empty((len(saved),paths,len(state_names)))
    dfs, fxs = np.empty((len(saved),paths,2)), np.empty((len(saved),paths))
    floor_counts = np.zeros(4,dtype=int)
    kernels = {}
    valuation_seconds,random_seconds = 0.,0.

    def record(step):
        nonlocal valuation_seconds
        record_start = time.perf_counter()
        t, out = float(times[step]), save_lookup.get(step)
        values = np.zeros_like(cumulative)
        values[:,:2] = spots*np.column_stack([np.ones(paths),fx])
        col = 2
        for c in range(2):
            conversion = 1 if c==0 else fx
            for maturity in government_maturities:
                if t < maturity-1e-9:
                    values[:,col] = conversion*market.rates[c].bond(market.curves[c],t,maturity,factors[:,c*m:(c+1)*m])
                col += 1
        for c in range(2):
            if t < bonds[c].maturity-1e-9:
                # Values needed only for surviving issuers; avoid needless pricing after default.
                mask = alive[:,c]
                values[mask,-2+c] = corporate_value(bonds[c],market.credits[c],market.rates[c],market.curves[c],
                    t,factors[mask,c*m:(c+1)*m],np.maximum(z[mask,c],0),recovery_intervals)*(1 if c==0 else fx[mask])
        current_gains = discount[:,0,None]*values+cumulative
        reported_v = np.abs(v) if heston_scheme=="reflection" else np.maximum(v,0)
        current_states = np.column_stack([spots[:,0],spots[:,1]*fx,reported_v,factors,np.maximum(z,0),fx])
        if out is not None:
            gains[out],states[out] = current_gains,current_states
            dfs[out],fxs[out] = discount,fx
            if store_security_values:
                security_values[out],distributions[out] = values,cumulative
        valuation_seconds += time.perf_counter()-record_start

    record(0)
    for step,(t,u) in enumerate(zip(times[:-1],times[1:])):
        h = float(u-t)
        key = round(h,12)
        if key not in kernels:
            covariance,root = joint_gaussian_kernel(h,r,rate_ids,a,sig)
            kernels[key] = root
        random_start = time.perf_counter()
        innovations = source.normals(step)@kernels[key].T
        random_seconds += time.perf_counter()-random_start
        dw = innovations[:,:d]
        prior = factors.copy()
        if scheme == "exact":
            factor_integrals = prior*B(a,h)+force*(h-B(a,h))/a+innovations[:,d+2*m:]
            factors = prior*np.exp(-a*h)+force*B(a,h)+innovations[:,d:d+2*m]
        else:
            factor_integrals = prior*h
            factors = prior+(-a*prior+force)*h+sig*dw[:,rate_ids]
        integrals = np.column_stack([market.rates[c].shift_integral(market.curves[c],float(t),float(u))
                       +factor_integrals[:,c*m:(c+1)*m].sum(axis=1) for c in range(2)])
        old_discount, old_fx = discount.copy(),fx.copy()
        discount *= np.exp(-integrals)
        fx *= np.exp(integrals[:,0]-integrals[:,1]-.5*market.fx_vol**2*h+market.fx_vol*dw[:,fx_id])
        for c,label in enumerate(("local","foreign")):
            eq = market.equities[c]
            sid,vid = names.index("S_"+label),names.index("v_"+label)
            positive = np.abs(v[:,c]) if heston_scheme=="reflection" else np.maximum(v[:,c],0)
            rootv = np.sqrt(positive)
            quanto_s = 0 if c==0 else -market.fx_vol*r[sid,fx_id]*rootv
            quanto_v = 0 if c==0 else -market.fx_vol*r[vid,fx_id]*eq.xi*rootv
            spots[:,c] *= np.exp(integrals[:,c]+(quanto_s-eq.dividend-.5*positive)*h+rootv*dw[:,sid])
            base_variance = positive if heston_scheme=="reflection" else v[:,c]
            v[:,c] = base_variance+(eq.kappa*(eq.theta-positive)+quanto_v)*h+eq.xi*rootv*dw[:,vid]
            cumulative[:,c] += discount[:,0]*spots[:,c]*np.expm1(eq.dividend*h)*(1 if c==0 else fx)
            cr = market.credits[c]
            zp = np.maximum(z[:,c],0)
            intensity = cr.intensity_multiplier(float(t+1e-10))*zp
            next_hazard = hazard[:,c]+intensity*h
            event = alive[:,c] & (next_hazard >= thresholds[:,c])
            fraction = np.divide(thresholds[:,c]-hazard[:,c],intensity*h,
                                  out=np.ones(paths),where=intensity>0)
            fraction = np.clip(fraction,0,1)  # event localization, not probability clipping
            if np.any(event):
                discount_tau = old_discount[event,0]*np.exp(-fraction[event]*integrals[event,0])
                fx_tau = old_fx[event]*np.exp(fraction[event]*np.log(fx[event]/old_fx[event]))
                cumulative[event,-2+c] += discount_tau*bonds[c].par*cr.recovery*(1 if c==0 else fx_tau)
                default_times[event,c] = t+fraction[event]*h
            alive[event,c] = False
            hazard[:,c] = next_hazard
            cid = names.index("credit_"+label)
            z[:,c] += cr.kappa*(1-zp)*h+cr.nu*np.sqrt(zp)*dw[:,cid]
            if any(abs(u-cf)<1e-9 for cf in bonds[c].times):
                amount = bonds[c].par*bonds[c].coupon/bonds[c].frequency
                if abs(u-bonds[c].maturity)<1e-9:
                    amount += bonds[c].par
                cumulative[:,-2+c] += alive[:,c]*discount[:,0]*amount*(1 if c==0 else fx)
            # No issuer events or additional recovery after maturity.
            if u >= bonds[c].maturity-1e-9:
                alive[:,c] = False
            for j,maturity in enumerate(government_maturities):
                if abs(u-maturity)<1e-9:
                    cumulative[:,2+c*len(government_maturities)+j] += discount[:,0]*(1 if c==0 else fx)
        floor_counts += [np.count_nonzero(v[:,0]<0),np.count_nonzero(v[:,1]<0),np.count_nonzero(z[:,0]<0),np.count_nonzero(z[:,1]<0)]
        if step+1 in save_lookup:
            record(step+1)
    return Paths(times[saved],np.arange(len(saved)),assets,state_names,
                 gains,states,dfs,fxs,default_times,
                 {"negative_raw_euler_counts":floor_counts.tolist(),"internal_steps":len(times)-1,
                  "scheme":scheme,"heston_scheme":heston_scheme,"random":"pseudo",
                  "rate_innovation_dimension":gaussian_dim,"seed":seed,
                  "recovery_intervals_per_year":recovery_intervals,
                  "timings":{"total_seconds":time.perf_counter()-simulation_start,
                    "reporting_and_valuation_seconds":valuation_seconds,"random_and_correlation_seconds":random_seconds,
                    "evolution_and_setup_seconds":time.perf_counter()-simulation_start-valuation_seconds-random_seconds}},
                 values=security_values,discounted_distributions=distributions)
