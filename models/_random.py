"""Internal seed/path/step shock addressing; shared by both simulators."""
import numpy as np
from scipy.special import ndtri


def indexed_uniform(seed, path_ids, step, dimensions):
    # SplitMix64 counter mixer. Fixed addressing, no data-dependent consumption.
    p = np.asarray(path_ids,dtype=np.uint64)[:,None]
    d = np.arange(dimensions,dtype=np.uint64)[None,:]
    with np.errstate(over="ignore"):
        x = (p*np.uint64(0xD2B74407B1CE6E93) + d*np.uint64(0xCA5A826395121157)
             + np.uint64(seed)*np.uint64(0x9E3779B97F4A7C15)
             + np.uint64(step+1)*np.uint64(0xA24BAED4963EE407))
        x = (x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x = (x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x ^= x>>np.uint64(31)
    return np.minimum(((x>>np.uint64(11)).astype(float)+.5)/2**53,np.nextafter(1.,0.))



class RandomSource:
    """Pseudo-random shocks with fixed addresses, independent of batch size."""

    def __init__(self, seed, path_ids, times, gaussian_dimensions, issuers=2):
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed = seed
        self.path_ids = np.asarray(path_ids)
        self.steps = len(times)
        self.gaussian_dimensions = gaussian_dimensions
        self.issuers = issuers

    def normals(self, step):
        return ndtri(indexed_uniform(self.seed, self.path_ids, step, self.gaussian_dimensions))

    def default_uniforms(self):
        return indexed_uniform(self.seed, self.path_ids, self.steps + 17, self.issuers)
