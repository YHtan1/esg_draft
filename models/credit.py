"""Representative coupon bonds, CIR intensity, and recovery valuation."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .rates import B


@dataclass(frozen=True)
class Credit:
    knots: tuple = (0., 5., 10., 40.)
    multipliers: tuple = (.015, .02, .025)
    kappa: float = .7
    nu: float = 0.0
    recovery: float = .4
    liquidity: float = 0.0
    rating: str = "synthetic-A"

    def __post_init__(self):
        if (len(self.knots) != len(self.multipliers)+1 or self.knots[0] != 0
                or np.any(np.diff(self.knots) <= 0) or min(self.multipliers) < 0
                or self.kappa <= 0 or self.nu < 0 or not 0 <= self.recovery <= 1
                or self.liquidity < 0 or not np.all(np.isfinite(
                    self.knots+self.multipliers+(self.kappa,self.nu,self.recovery,self.liquidity)))):
            raise ValueError("Invalid credit curve/CIR/recovery assumptions")

    def intensity_multiplier(self, t):
        return self.multipliers[min(np.searchsorted(self.knots,t,side="right")-1,len(self.multipliers)-1)]

    @lru_cache(maxsize=131072)
    def affine(self, t, u):
        if t < 0 or u < t or u > self.knots[-1]+1e-10:
            raise ValueError("Credit survival outside fitted horizon")
        cuts = [t]+[x for x in self.knots if t < x < u]+[u]
        log_a, b = 0.0, 0.0
        for left, right in reversed(list(zip(cuts[:-1], cuts[1:]))):
            h, a = right-left, self.intensity_multiplier((left+right)/2)
            if self.nu < 1e-10:
                integral = b*B(self.kappa,h)+a*(h-B(self.kappa,h))/self.kappa
                b = b*np.exp(-self.kappa*h)+a*B(self.kappa,h)
            else:
                gamma = np.sqrt(self.kappa**2+2*self.nu**2*a)
                # Stable positive root avoids cancellation when a is small.
                plus, minus = 2*a/(self.kappa+gamma), -(self.kappa+gamma)/self.nu**2
                y0 = (b-plus)/(b-minus)
                y1 = y0*np.exp(-gamma*h)
                integral = plus*h+2/self.nu**2*(np.log1p(-y1)-np.log1p(-y0))
                b = (plus-y1*minus)/(1-y1)
            log_a -= self.kappa*integral  # long-run Z mean is one
        return float(log_a), float(b)

    def survival(self, t, u, z=1.):
        a,b = self.affine(float(t),float(u))
        return np.exp(a-b*np.maximum(z,0))


@dataclass(frozen=True)
class CorporateBond:
    name: str
    currency: str
    maturity: float
    coupon: float = .045
    frequency: int = 1
    par: float = 100.0

    def __post_init__(self):
        if (self.maturity <= 0 or self.frequency not in (1,2,4) or self.par <= 0
                or self.coupon < 0 or not np.isfinite(self.maturity+self.par+self.coupon)
                or abs(self.maturity*self.frequency-round(self.maturity*self.frequency)) > 1e-9):
            raise ValueError("Use positive, regular coupon schedules")

    @property
    def times(self):
        return np.arange(1,round(self.maturity*self.frequency)+1,dtype=float)/self.frequency


def corporate_value(bond, credit, rate_model, curve, t, factors, z, recovery_intervals=24):
    """Alive ex-coupon value; rates/credit must be independent.

    Recovery uses midpoint discounting against exact affine default probabilities.
    Its deterministic quadrature error is separately tested, not hidden in MC error.
    Liquidity is a valuation spread only; scenarios require liquidity=0 for gains tests.
    """
    result = np.zeros(np.broadcast_shapes(np.shape(z), np.shape(factors)[:-1]))
    for u in bond.times:
        if u > t+1e-9:
            amount = bond.par*bond.coupon/bond.frequency + (bond.par if abs(u-bond.maturity)<1e-9 else 0)
            result += amount*rate_model.bond(curve,t,float(u),factors)*credit.survival(t,float(u),z)*np.exp(-credit.liquidity*(u-t))
    if t >= bond.maturity-1e-9 or credit.recovery == 0:
        return result
    # Include all hazard knots in quadrature partitions.
    count = max(1, int(np.ceil((bond.maturity-t)*recovery_intervals)))
    nodes = np.unique(np.r_[np.linspace(t,bond.maturity,count+1),
                            [u for u in credit.knots if t < u < bond.maturity]])
    left_survival = np.ones_like(result)
    for left,right in zip(nodes[:-1],nodes[1:]):
        right_survival = credit.survival(t,float(right),z)
        mid = float((left+right)/2)
        result += bond.par*credit.recovery*rate_model.bond(curve,t,mid,factors)*(left_survival-right_survival)*np.exp(-credit.liquidity*(mid-t))
        left_survival = right_survival
    return result

