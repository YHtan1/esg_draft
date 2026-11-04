"""Validated, immutable simulation inputs and named Brownian driver registry."""
from dataclasses import dataclass
import numpy as np
from .curves import DiscountCurve
from .rates import GaussianRates, psd_root
from .equity import Heston
from .credit import Credit


def driver_names(model="g2"):
    if model not in ("hw","g2"):
        raise ValueError("Rate model must be hw or g2")
    return tuple(["S_local","v_local","x_local"]+(["y_local"] if model=="g2" else [])
                 +["S_foreign","v_foreign","x_foreign"]+(["y_foreign"] if model=="g2" else [])
                 +["credit_local","credit_foreign","FX"])


@dataclass(frozen=True)
class Market:
    domestic: str
    foreign: str
    curves: tuple
    rates: tuple
    equities: tuple
    credits: tuple
    correlation: tuple
    fx_spot: float = 35.0
    fx_vol: float = .12
    snapshot: str = "research-inputs"
    synthetic: bool = False

    def __post_init__(self):
        if self.domestic == self.foreign or len(self.rates) != 2 or len(self.curves)!=2 or len(self.equities)!=2 or len(self.credits)!=2:
            raise ValueError("Supply two distinct currencies and two models per component")
        if not np.isfinite(self.fx_spot+self.fx_vol) or self.fx_spot <= 0 or self.fx_vol < 0:
            raise ValueError("Invalid FX assumptions")
        if len(self.rates[0].mean_reversions) != len(self.rates[1].mean_reversions):
            raise ValueError("Both currencies must use the same rate model")
        names, r = self.drivers, np.asarray(self.correlation)
        if r.shape != (len(names),len(names)) or not np.allclose(np.diag(r),1,atol=1e-12,rtol=0):
            raise ValueError("Correlation dimensions/diagonal do not match named registry")
        psd_root(r)
        for c,label in enumerate(("local","foreign")):
            if abs(r[names.index("S_"+label),names.index("v_"+label)]-self.equities[c].rho)>1e-12:
                raise ValueError("Heston rho must agree with driver matrix; never apply it twice")
            ids = [names.index("x_"+label)]+([names.index("y_"+label)] if self.model=="g2" else [])
            if not np.allclose(r[np.ix_(ids,ids)],self.rates[c].correlation,atol=1e-12,rtol=0):
                raise ValueError("Rate correlation inconsistent with Gaussian bond model")
            credit_id = names.index("credit_"+label)
            if np.any(np.abs(np.delete(r[credit_id],credit_id)) > 1e-12):
                raise ValueError("Initial credit implementation requires independence from every other driver")
            if self.credits[c].liquidity != 0:
                raise ValueError("Joint gains simulation requires zero liquidity spread")

    @property
    def model(self):
        return "hw" if len(self.rates[0].mean_reversions)==1 else "g2"

    @property
    def drivers(self):
        return driver_names(self.model)


def make_grid(horizon, internal_steps=12, reporting_steps=12, cashflow_times=()):
    if horizon <= 0 or internal_steps < 1 or reporting_steps < 1 or not np.isfinite(horizon):
        raise ValueError("Positive horizon and frequencies required")
    def dates(freq):
        return np.r_[np.arange(int(np.floor(horizon*freq))+1)/freq,horizon]
    cashflows = np.asarray(cashflow_times,dtype=float)
    if np.any(~np.isfinite(cashflows)) or np.any(cashflows < 0):
        raise ValueError("Invalid cash-flow times")
    reporting = np.unique(np.round(dates(reporting_steps),12))
    times = np.unique(np.round(np.r_[dates(internal_steps),reporting,cashflows[cashflows<=horizon]],12))
    return times, np.searchsorted(times,reporting)


def market_from_dict(data):
    allowed = {"domestic","foreign","curves","rates","equities","credits","correlation","fx_spot","fx_vol","snapshot","synthetic"}
    if set(data)-allowed:
        raise ValueError("Unknown market fields: "+str(set(data)-allowed))
    d = dict(data)
    d["curves"] = tuple(DiscountCurve(tuple(x["times"]),tuple(x["discounts"]),x.get("extrapolation","flat_forward")) for x in d["curves"])
    d["rates"] = tuple(GaussianRates(tuple(x["mean_reversions"]),tuple(x["volatilities"]),tuple(tuple(r) for r in x["correlation"])) for x in d["rates"])
    d["equities"] = tuple(Heston(**x) for x in d["equities"])
    d["credits"] = tuple(Credit(**{k:tuple(v) if k in ("knots","multipliers") else v for k,v in x.items()}) for x in d["credits"])
    d["correlation"] = tuple(tuple(x) for x in d["correlation"])
    return Market(**d)


def load_market(directory=None, model="g2"):
    """Read dated observed curves/spots plus explicitly assumed dynamics.

    No network access, calibration, or synthetic fallback occurs here.
    """
    from pathlib import Path
    import json
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parents[1]/"data"
    def read(name):
        return json.loads((directory/name).read_text())
    assumptions = read("assumptions.json")["models"]
    if model not in assumptions:
        raise ValueError("Rate model must be hw or g2")
    manifest = read("snapshot.json")
    data = [read("thai_curve.json"), read("us_curve.json")]
    indices = read("stock_indices.json")
    inputs = assumptions[model]
    inputs["curves"] = []
    for source in data:
        t = np.array([row["maturity"] for row in source["nodes"]])
        zero = np.array([row["zero_cc"] for row in source["nodes"]])
        inputs["curves"].append({"times":tuple(np.r_[0.,t]),"discounts":tuple(np.r_[1.,np.exp(-zero*t)])})
    for equity, observation in zip(inputs["equities"], indices["observations"]):
        equity["spot"] = observation["close"]
        equity["dividend"] = observation.get("trailing_dividend_yield", equity["dividend"])
    inputs.update(snapshot="OBSERVED-CURVES-INDEX-SPOTS-"+manifest["snapshot_date"], synthetic=False)
    return market_from_dict(inputs), {"manifest":manifest,"curves":data,"indices":indices,"us_par":read("us_par_yields.json")}
