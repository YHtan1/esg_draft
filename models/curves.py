"""Published zero nodes converted to log-linear discount curves."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DiscountCurve:
    times: tuple
    discounts: tuple
    extrapolation: str = "flat_forward"

    def __post_init__(self):
        if (len(self.times) != len(self.discounts) or len(self.times) < 2
                or self.times[0] != 0 or self.discounts[0] != 1
                or not np.all(np.isfinite(self.times + self.discounts))
                or np.any(np.diff(self.times) <= 0) or min(self.discounts) <= 0
                or self.extrapolation not in ("flat_forward", "error")):
            raise ValueError("Invalid discount curve")

    def log_discount(self, t):
        t = np.asarray(t, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0):
            raise ValueError("Discount times must be finite and nonnegative")
        if self.extrapolation == "error" and np.any(t > self.times[-1] + 1e-12):
            raise ValueError("Curve extrapolation disabled")
        logs = np.log(self.discounts)
        result = np.interp(t, self.times, logs)
        slope = (logs[-1] - logs[-2]) / (self.times[-1] - self.times[-2])
        return np.where(t > self.times[-1], logs[-1] + slope * (t-self.times[-1]), result)

    def discount(self, t):
        return np.exp(self.log_discount(t))

    def forward(self, t):
        self.log_discount(t)  # enforce the same domain/extrapolation policy
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times)-2)
        return -np.diff(np.log(self.discounts))[idx] / np.diff(self.times)[idx]

