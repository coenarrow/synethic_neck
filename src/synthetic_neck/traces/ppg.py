"""Finger photoplethysmogram (pulse oximeter pleth, arbitrary units, systole up). The pulse is the ABP Windkessel
shape with the finger row, its foot delayed by the radial-to-finger transit. It is scaled to the drawn height about
a zero mean, then the pulse height and the baseline both fall at end-inspiration and white noise is added."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .abp import abp_shape
from .ecg import t_wave_end
from .priors import Range


@dataclass(frozen=True)
class PpgConfig:
    amplitude: Range
    radial_to_finger_s: Range
    am_fraction: Range
    wander: Range
    noise: Range


def ppg(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, duration_s: float,
        amplitude: float, pre_ejection_s: float, transit_s: float, am_fraction: float, wander: float, noise: float,
        rng: np.random.Generator) -> np.ndarray:
    """Finger PPG on the grid `t`. Ejection runs from `pre_ejection_s` after the R wave to the end of the T wave,
    the same draw the ABP uses; `transit_s` is the whole aortic-to-finger transit. The finger shape is rescaled
    over 0 <= t < duration_s to peak-to-trough `amplitude` about a zero mean, then
    PPG = PPG0 (1 - am_fraction x) - wander x + noise eps, with x = `resp`."""
    r = np.asarray(r_times, dtype=np.float64)
    ejection = t_wave_end(r, heart_rate_bpm) - r[:-1] - pre_ejection_s
    feet = r[:-1] + pre_ejection_s + transit_s
    s = abp_shape(t, feet, ejection, "finger")
    win = (t >= 0.0) & (t < duration_s)
    s = (s - s[win].min()) / (s[win].max() - s[win].min())
    base = amplitude * (s - s[win].mean())
    resp = np.asarray(resp, dtype=np.float64)
    return base * (1.0 - am_fraction * resp) - wander * resp + noise * rng.standard_normal(base.shape)
