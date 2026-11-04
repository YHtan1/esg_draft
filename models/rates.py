"""Gaussian short rates: joint endpoints, time integrals and all Brownian increments."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import quad


def B(a, t):
    return -np.expm1(-a*np.asarray(t))/a


def psd_root(matrix, tolerance=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("Covariance/correlation must be finite and symmetric")
    vals, vectors = np.linalg.eigh(matrix)
    if vals.min() < -tolerance * max(1.0, vals.max()):
        raise ValueError("Matrix is not positive semidefinite; no repair performed")
    # Only floating-point eigensolver roundoff at the PSD boundary is zeroed.
    return vectors * np.sqrt(np.maximum(vals, 0.0))


@dataclass(frozen=True)
class GaussianRates:
    mean_reversions: tuple
    volatilities: tuple
    correlation: tuple

    def __post_init__(self):
        m = len(self.mean_reversions)
        if (m not in (1, 2) or len(self.volatilities) != m
                or min(self.mean_reversions) <= 0 or min(self.volatilities) < 0
                or not np.all(np.isfinite(self.mean_reversions + self.volatilities))):
            raise ValueError("Use positive mean reversions and nonnegative volatilities")
        r = np.asarray(self.correlation)
        if r.shape != (m, m) or not np.allclose(np.diag(r), 1):
            raise ValueError("Invalid rate factor correlation")
        psd_root(r)

    @lru_cache(maxsize=8192)
    def integral_variance(self, t):
        if t <= 0:
            return 0.0
        a, sig, r = self.mean_reversions, self.volatilities, np.array(self.correlation)
        return sum(sig[i]*sig[j]*r[i,j]*quad(lambda u: B(a[i],u)*B(a[j],u), 0,t,
                   epsabs=1e-13)[0] for i in range(len(a)) for j in range(len(a)))

    def shift_integral(self, curve, t, u):
        return (curve.log_discount(t)-curve.log_discount(u)
                + .5*(self.integral_variance(float(u))-self.integral_variance(float(t))))

    def bond(self, curve, t, maturity, factors):
        if maturity < t-1e-10:
            raise ValueError("Bond maturity before valuation")
        tau = max(0.0, maturity-t)
        adjustment = .5*(self.integral_variance(tau)-self.integral_variance(float(maturity))
                         +self.integral_variance(float(t)))
        loading = np.array([B(a,tau) for a in self.mean_reversions])
        return np.exp(curve.log_discount(maturity)-curve.log_discount(t)+adjustment-factors@loading)


def joint_gaussian_kernel(dt, correlation, rate_indices, mean_reversions, volatilities):
    """Outputs [dW(all named drivers), endpoint noise, integral noise].

    Gaussian quadrature integrates deterministic kernels, including rate/equity/FX
    cross covariances. 32 points is far beyond float64 precision on supported steps.
    """
    if not np.isfinite(dt) or dt<=0 or dt>1.:
        raise ValueError("Joint Gaussian kernel supports positive steps up to one year")
    r = np.asarray(correlation)
    d, m = len(r), len(rate_indices)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    u, weights = (nodes+1)*dt/2, weights*dt/2
    kernels = np.ones((d+2*m, len(u)))
    drivers = list(range(d))+list(rate_indices)*2
    for j, (a, sig) in enumerate(zip(mean_reversions, volatilities)):
        kernels[d+j] = sig*np.exp(-a*u)
        kernels[d+m+j] = sig*B(a,u)
    covariance = (kernels*weights)@kernels.T * r[np.ix_(drivers, drivers)]
    return covariance, psd_root(covariance)
