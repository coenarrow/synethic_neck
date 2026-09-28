"""Prior distribution types used by the trace configs. Each has `draw(rng)` returning one float."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Range:
    """A uniform [lo, hi] prior; `draw` returns one value."""
    lo: float
    hi: float

    def __post_init__(self):
        if self.hi < self.lo:
            raise ValueError(f"Range hi {self.hi} < lo {self.lo}")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class Normal:
    """A normal prior with mean `mean` and sd `sd`, clipped to [lo, hi] (None = unbounded)."""
    mean: float
    sd: float
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self):
        if self.sd < 0:
            raise ValueError(f"Normal sd {self.sd} < 0")
        if self.lo is not None and self.hi is not None and self.hi < self.lo:
            raise ValueError(f"Normal hi {self.hi} < lo {self.lo}")

    def draw(self, rng: np.random.Generator) -> float:
        x = rng.normal(self.mean, self.sd)
        return float(np.clip(x, -np.inf if self.lo is None else self.lo, np.inf if self.hi is None else self.hi))


@dataclass(frozen=True)
class IntRange:
    """A uniform integer prior on [lo, hi] inclusive; `draw` returns one int."""
    lo: int
    hi: int

    def __post_init__(self):
        if self.hi < self.lo:
            raise ValueError(f"IntRange hi {self.hi} < lo {self.lo}")

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.lo, self.hi + 1))
