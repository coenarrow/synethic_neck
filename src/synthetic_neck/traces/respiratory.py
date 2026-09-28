"""Respiration: the shared clock. A slowly drifting rate is integrated to a phase, and the waveform is its sine.
Every other trace reads this waveform for its respiratory coupling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .priors import Range


@dataclass(frozen=True)
class RespConfig:
    rate_bpm: Range
    rate_drift_frac: Range
    rate_drift_tau_s: Range


def respiratory_rate(t: np.ndarray, mean_bpm: float, drift_frac: float, tau_s: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Instantaneous respiratory rate (breaths/min) on `t`: the mean times (1 + d), where d is slow Gaussian
    wander with standard deviation `drift_frac` and correlation time `tau_s` (an Ornstein-Uhlenbeck process).
    d is clipped to +/-0.5 so the rate stays positive. drift_frac == 0 gives a constant rate."""
    t = np.asarray(t, dtype=np.float64)
    if drift_frac <= 0 or t.size < 2:
        return np.full(t.shape, float(mean_bpm))
    dt = float(np.median(np.diff(t)))
    a = np.exp(-dt / tau_s)
    kick = drift_frac * np.sqrt(1.0 - a * a) * rng.standard_normal(t.size)
    kick[0] = drift_frac * rng.standard_normal()          # start from the stationary distribution
    d = lfilter([1.0], [1.0, -a], kick)                    # d[i] = a d[i-1] + kick[i]
    return mean_bpm * (1.0 + np.clip(d, -0.5, 0.5))


def respiration(t: np.ndarray, rate_bpm: float | np.ndarray, phase_rad: float) -> np.ndarray:
    """Respiratory waveform x(t) in [-1, 1]; +1 is end-inspiration. `rate_bpm` may be a scalar or a series on
    `t`; the phase is the integral of the rate, so a drifting rate gives smoothly varying breath lengths."""
    t = np.asarray(t, dtype=np.float64)
    f_hz = np.broadcast_to(np.asarray(rate_bpm, dtype=np.float64) / 60.0, t.shape)
    cycles = np.concatenate([[0.0], np.cumsum(0.5 * (f_hz[1:] + f_hz[:-1]) * np.diff(t))])
    return np.sin(2 * np.pi * cycles + phase_rad)
