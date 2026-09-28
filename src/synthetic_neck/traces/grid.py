"""The generation grid shared by every trace.

Every trace is generated on a fixed 1 kHz grid (TRACE_HZ) padded by PAD_S on both sides so edge beats and
breaths are complete, then cropped to [0, duration) and resampled to the config's sample rate with `resample`.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

TRACE_HZ = 1000.0
PAD_S = 2.0


def time_grid(duration_s: float) -> np.ndarray:
    """Generation grid: [-PAD_S, duration_s + PAD_S) at TRACE_HZ."""
    n = int(round((duration_s + 2 * PAD_S) * TRACE_HZ))
    return -PAD_S + np.arange(n) / TRACE_HZ


def crop(t: np.ndarray, x: np.ndarray, duration_s: float) -> np.ndarray:
    """The part of `x` with 0 <= t < duration_s."""
    return x[(t >= 0.0) & (t < duration_s)]


def resample(x: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Anti-aliased rational resampling (polyphase FIR) from `from_hz` to `to_hz`."""
    if from_hz == to_hz:
        return np.asarray(x, dtype=np.float64).copy()
    ratio = Fraction(to_hz / from_hz).limit_denominator(10_000)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)
