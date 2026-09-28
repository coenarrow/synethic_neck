"""Arterial blood pressure at the brachial and radial sites. Per beat the aortic valve opens a pre-ejection
period after the R wave and closes at the end of the T wave; a two-Gaussian ejection flow between those events
drives a three-element Windkessel, and the shape is scaled to the drawn mean and pulse pressure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .ecg import t_wave_end
from .priors import Normal, Range

TAU_D_S = 1.5                       # Windkessel diastolic run-off time constant (s)

# Shape constants per site: (c_s, b_s, a_s, c_d, b_d, a_d, z_c). The ejection flow per beat is two Gaussians in time
# since the foot, systolic ejection then the dicrotic wave, with centres c and widths b as fractions of that beat's
# ejection time and relative amplitudes a. z_c is the characteristic impedance of the three-element Windkessel:
# pressure = z_c * flow + compliance pressure. Initial values, tuned by eye.
# The finger row is the volume pulse the PPG reads: a lower impedance and wider ejection round off the systolic
# peak, and the reflected wave is later and blends into the run-off. The carotid row is the central pressure the
# neck sees: the reflected wave arrives in late systole, before the valve closes, so the beat has a first peak,
# a lower second peak and then the incisura at the end of ejection.
ABP_SHAPE = {"carotid": (0.25, 0.10, 1.0, 0.75, 0.12, 0.30, 0.08),
             "brachial": (0.25, 0.10, 1.0, 1.15, 0.08, 0.25, 0.05),
             "radial": (0.22, 0.08, 1.0, 1.30, 0.10, 0.30, 0.08),
             "finger": (0.30, 0.14, 1.0, 1.40, 0.14, 0.30, 0.02)}


@dataclass(frozen=True)
class AbpConfig:
    map_mmhg: Normal
    pulse_pressure_mmhg: Normal
    pre_ejection_s: Range
    aortic_to_brachial_s: Range
    brachial_to_radial_s: Range
    radial_amplification: Range
    resp_swing_mmhg: Range
    noise_mmhg: Range


def ejection_flow(t: np.ndarray, 
                  feet: np.ndarray, 
                  ejection_s: np.ndarray, 
                  site: str) -> np.ndarray:
    """Flow into the Windkessel on the grid `t`: per beat, two Gaussians in time since the foot at fractions of the
    beat's ejection time (ABP_SHAPE[site])."""
    t = np.asarray(t, dtype=np.float64)
    c_s, b_s, a_s, c_d, b_d, a_d, _ = ABP_SHAPE[site]
    q = np.zeros_like(t)
    for f, ts in zip(feet, ejection_s):
        for c, b, a in ((c_s, b_s, a_s), (c_d, b_d, a_d)):
            centre, width = f + c * ts, b * ts
            lo, hi = np.searchsorted(t, (centre - 5 * width, centre + 5 * width))
            q[lo:hi] += a * np.exp(-0.5 * ((t[lo:hi] - centre) / width) ** 2)
    return q


def windkessel(t: np.ndarray, 
               q: np.ndarray, 
               tau_s: float = TAU_D_S) -> np.ndarray:
    """Compliance pressure from flow `q`, from rest: ds/dt = -s / tau_s + q, i.e. s_i = exp(-dt / tau_s) s_{i-1} + dt q_i."""
    t, q = np.asarray(t, dtype=np.float64), np.asarray(q, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    return lfilter([dt], [1.0, -np.exp(-dt / tau_s)], q)


def abp_shape(t: np.ndarray, 
              feet: np.ndarray, 
              ejection_s: np.ndarray, 
              site: str,
              warmup_s: float = 5 * TAU_D_S) -> np.ndarray:
    """Unscaled pressure shape at `site`: three-element Windkessel, z_c * flow + compliance pressure. The grid is
    extended backwards by `warmup_s` with virtual beats at the first interval, so the Windkessel is in steady
    state by the start of `t` (the transient has decayed by exp(-warmup_s / TAU_D_S))."""
    t, feet, ejection_s = (np.asarray(v, dtype=np.float64) for v in (t, feet, ejection_s))
    dt = float(np.median(np.diff(t)))
    rr = float(np.median(np.diff(feet))) if feet.size > 1 else 1.0
    n_virtual = int(np.ceil(warmup_s / rr)) + 1
    virtual_feet = feet[0] - rr * np.arange(n_virtual, 0, -1)
    n_pad = int(np.ceil((t[0] - virtual_feet[0] + rr) / dt))
    t_ext = np.concatenate([t[0] - dt * np.arange(n_pad, 0, -1), t])
    q = ejection_flow(t_ext, np.concatenate([virtual_feet, feet]),
                      np.concatenate([np.full(n_virtual, ejection_s[0]), ejection_s]), site)
    return (ABP_SHAPE[site][-1] * q + windkessel(t_ext, q))[n_pad:]


def abp(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, duration_s: float,
        map_mmhg: float, pulse_pressure_mmhg: float, pre_ejection_s: float, transit_s: float, site: str,
        amplification: float, resp_swing_mmhg: float, noise_mmhg: float, rng: np.random.Generator) -> np.ndarray:
    """Arterial pressure (mmHg) at `site` on the grid `t`. Per beat the aortic valve opens `pre_ejection_s` after
    the R wave and closes at the end of the T wave; the foot is the opening delayed by `transit_s`. The ejection
    flow drives a three-element Windkessel (`abp_shape`); the shape is rescaled over 0 <= t < duration_s so its
    mean is `map_mmhg` and its range `amplification * pulse_pressure_mmhg`. Then
    ABP = ABP0 - resp_swing_mmhg x + noise_mmhg eps."""
    r = np.asarray(r_times, dtype=np.float64)
    ejection = t_wave_end(r, heart_rate_bpm) - r[:-1] - pre_ejection_s
    feet = r[:-1] + pre_ejection_s + transit_s
    s = abp_shape(t, feet, ejection, site)
    win = (t >= 0.0) & (t < duration_s)
    s = (s - s[win].min()) / (s[win].max() - s[win].min())
    base = map_mmhg + amplification * pulse_pressure_mmhg * (s - s[win].mean())
    resp = np.asarray(resp, dtype=np.float64)
    return base - resp_swing_mmhg * resp + noise_mmhg * rng.standard_normal(base.shape)
