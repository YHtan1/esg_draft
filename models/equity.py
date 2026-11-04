"""Heston parameters, analytic QuantLib prices, and four simulation methods.

QE methods are standalone constant-rate benchmarks, not hybrid rate/FX schemes.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, ndtri
from ._random import indexed_uniform

METHODS = ("full_truncation", "reflection", "qe", "qe_martingale")


@dataclass(frozen=True)
class Heston:
    spot: float = 100.0
    v0: float = .04
    kappa: float = 1.5
    theta: float = .04
    xi: float = .35
    rho: float = -.65
    dividend: float = .02

    def __post_init__(self):
        values = (self.spot, self.v0, self.kappa, self.theta, self.xi, self.rho, self.dividend)
        if (not np.all(np.isfinite(values)) or self.spot <= 0 or min(self.v0,self.theta,self.xi) < 0
                or self.kappa <= 0 or abs(self.rho) > 1):
            raise ValueError("Invalid Heston parameters")

    @property
    def feller_margin(self):
        return 2*self.kappa*self.theta-self.xi**2


def heston_prices(params, quotes, rate=.03):
    import QuantLib as ql
    # Restore global QuantLib evaluation date: workers must own separate processes.
    old = ql.Settings.instance().evaluationDate
    today = ql.Date(4,11,2026)
    try:
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        r = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
        q = ql.YieldTermStructureHandle(ql.FlatForward(today, params.dividend, dc))
        model = ql.HestonModel(ql.HestonProcess(r,q,ql.QuoteHandle(ql.SimpleQuote(params.spot)),
                    params.v0,params.kappa,params.theta,max(params.xi,1e-8),params.rho))
        engine = ql.AnalyticHestonEngine(model, 144)
        result = []
        for quote in quotes:
            expiry = today+int(round(365*quote["maturity"]))
            option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call,quote["strike"]),
                                      ql.EuropeanExercise(expiry))
            option.setPricingEngine(engine)
            result.append(option.NPV())
        return np.array(result)
    finally:
        ql.Settings.instance().evaluationDate = old



def step_heston(params,spot,variance,dt,z,rate=.03,method="full_truncation"):
    """One step using two independent supplied normals, matching QuantLib order.

    Euler: z0 drives spot and rho*z0+sqrt(1-rho²)*z1 drives variance.
    QE: z1 drives variance, z0 is the orthogonal spot innovation.
    Reflection returns a raw endpoint; take abs at the NEXT step/reporting date.
    """
    spot,variance,z=np.asarray(spot),np.asarray(variance),np.asarray(z)
    if method not in METHODS or dt<=0 or spot.ndim!=1 or z.shape!=spot.shape+(2,) or variance.shape!=spot.shape:
        raise ValueError("Invalid Heston method, shapes or timestep")
    k,theta,xi,rho=params.kappa,params.theta,params.xi,params.rho
    if method in ("full_truncation","reflection"):
        v=np.abs(variance) if method=="reflection" else np.maximum(variance,0.)
        root=np.sqrt(v*dt)
        s=spot*np.exp((rate-params.dividend-.5*v)*dt+root*z[...,0])
        base=v if method=="reflection" else variance
        vnext=base+k*(theta-v)*dt+xi*root*(rho*z[...,0]+np.sqrt(1-rho*rho)*z[...,1])
        return s,vnext
    if np.any(variance<0):raise ValueError("QE requires nonnegative variance")
    decay=np.exp(-k*dt)
    if xi<1e-12:
        vnext=theta+(variance-theta)*decay
        integrated=theta*dt+(variance-theta)*(-np.expm1(-k*dt))/k
        return spot*np.exp((rate-params.dividend)*dt-.5*integrated+np.sqrt(np.maximum(integrated,0))*z[...,0]),vnext
    m=theta+(variance-theta)*decay
    s2=variance*xi*xi*decay/k*(1-decay)+theta*xi*xi/(2*k)*(1-decay)**2
    active=m>1e-16
    psi=np.divide(s2,m*m,out=np.zeros_like(m),where=active)
    k1=.5*dt*(k*rho/xi-.5)-rho/xi
    k2=.5*dt*(k*rho/xi-.5)+rho/xi
    k3=k4=.5*dt*(1-rho*rho)
    A=k2+.5*k4
    vnext=np.zeros_like(m)
    log_mgf=np.zeros_like(m)
    quadratic=active&(psi<1.5)
    if np.any(quadratic):
        p=np.maximum(psi[quadratic],1e-15)
        b2=2/p-1+np.sqrt(2/p*(2/p-1))
        a=m[quadratic]/(1+b2)
        vnext[quadratic]=a*(np.sqrt(b2)+z[quadratic,1])**2
        if method=="qe_martingale":
            denominator=1-2*A*a
            if np.any(denominator<=0):raise ValueError("QE moment condition violated; reduce timestep")
            log_mgf[quadratic]=A*b2*a/denominator-.5*np.log(denominator)
    exponential=active&~quadratic
    if np.any(exponential):
        p=(psi[exponential]-1)/(psi[exponential]+1)
        beta=(1-p)/m[exponential]
        u=np.minimum(ndtr(z[exponential,1]),np.nextafter(1.,0.))
        vnext[exponential]=np.where(u<=p,0.,np.log((1-p)/(1-u))/beta)
        if method=="qe_martingale":
            if np.any(beta<=A):raise ValueError("QE moment condition violated; reduce timestep")
            log_mgf[exponential]=np.log(p+beta*(1-p)/(beta-A))
    k0=-log_mgf-(k1+.5*k3)*variance if method=="qe_martingale" else np.full_like(m,-rho*k*theta*dt/xi)
    # The zero-variance, zero-long-run-variance state is absorbing.
    k0=np.where(active,k0,0.)
    s=spot*np.exp((rate-params.dividend)*dt+k0+k1*variance+k2*vnext+
                  np.sqrt(np.maximum(k3*variance+k4*vnext,0))*z[...,0])
    return s,vnext


@dataclass
class HestonPaths:
    times:np.ndarray
    spots:np.ndarray
    variances:np.ndarray
    discounted_total_return:np.ndarray
    call_payoffs:np.ndarray
    negative_raw_updates:int


def simulate_heston(params,paths=1000,horizon=10.,steps_per_year=12,seed=7,
                    method="full_truncation",rate=.03,keep_paths=True):
    steps=int(round(horizon*steps_per_year))
    if paths<2 or steps<1 or abs(steps/steps_per_year-horizon)>1e-10:
        raise ValueError("Positive integer-step horizon and >=2 paths required")
    times=np.linspace(0,horizon,steps+1);dt=horizon/steps
    spot=np.full(paths,params.spot);variance=np.full(paths,params.v0)
    count=0
    levels=np.empty((steps+1,paths)) if keep_paths else None
    vs=np.empty((steps+1,paths)) if keep_paths else None
    if keep_paths:levels[0],vs[0]=spot,variance
    for step in range(steps):
        z=ndtri(indexed_uniform(seed,np.arange(paths),step,2))
        spot,variance=step_heston(params,spot,variance,dt,z,rate,method)
        count+=int(np.count_nonzero(variance<0))
        if keep_paths:
            levels[step+1]=spot
            vs[step+1]=np.abs(variance) if method=="reflection" else np.maximum(variance,0)
    total_return=(levels if keep_paths else spot)*np.exp(-(rate-params.dividend)*(times[:,None] if keep_paths else horizon))/params.spot
    return HestonPaths(times,levels,vs,total_return,
                       np.exp(-rate*horizon)*np.maximum(spot-params.spot,0),count)

