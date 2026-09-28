"""Central venous (right atrial) pressure. Per beat the a, c and v waves and the x and y descents are Gaussians
in time, each placed by an event of the cardiac cycle read from the ECG: the a wave a delay after the P wave, the
c wave at the start of ejection, the x descent and v wave at fractions of the ejection time, and the y descent a
fixed time after the v wave. The sum is scaled to the drawn mean and peak-to-trough pulse."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ecg import p_wave_time, t_wave_end
from .priors import Normal, Range

# Gaussian centres and widths. The c wave, x descent and v wave are in units of the beat's ejection time, measured
# from its start (tricuspid closure to tricuspid opening spans systole plus isovolumic relaxation, hence v past 1).
# The a wave has a fixed width in seconds and sits at the drawn delay after the P wave; the y descent is a fixed
# time after the v wave since rapid filling does not shorten with rate. Initial values, tuned by eye
# against single-beat plots.
A_WIDTH_S = 0.05
C_CENTRE, C_WIDTH = 0.0, 0.06
X_CENTRE, X_WIDTH = 0.60, 0.20
V_CENTRE, V_WIDTH = 1.20, 0.18
Y_DELAY_S, Y_WIDTH_S = 0.15, 0.07


@dataclass(frozen=True)
class CvpConfig:
    mean_mmhg: Normal
    pulse_mmhg: Range
    c_to_a: Range
    v_to_a: Range
    x_to_a: Range
    y_to_a: Range
    p_to_a_s: Range
    resp_swing_mmhg: Range
    noise_mmhg: Range


def _add_gaussians(t: np.ndarray, s: np.ndarray, centres: np.ndarray, widths: np.ndarray, amp: float) -> None:
    for centre, width in zip(centres, widths):
        lo, hi = np.searchsorted(t, (centre - 5 * width, centre + 5 * width))
        s[lo:hi] += amp * np.exp(-0.5 * ((t[lo:hi] - centre) / width) ** 2)


def cvp_shape(t: np.ndarray, r_times: np.ndarray, heart_rate_bpm: float, pre_ejection_s: float, p_to_a_s: float,
              a_mmhg: float, c_mmhg: float, x_mmhg: float, v_mmhg: float, y_mmhg: float) -> np.ndarray:
    """Zero-based pressure shape (mmHg) on the grid `t`: the five Gaussians per beat with the given heights, the
    descents entered as positive depths. Ejection of each beat runs from `pre_ejection_s` after the R wave to
    the end of the T wave."""
    t, r = np.asarray(t, dtype=np.float64), np.asarray(r_times, dtype=np.float64)
    start = r[:-1] + pre_ejection_s
    ejection = t_wave_end(r, heart_rate_bpm) - start
    s = np.zeros_like(t)
    _add_gaussians(t, s, p_wave_time(r, heart_rate_bpm) + p_to_a_s, np.full(r.size - 1, A_WIDTH_S), a_mmhg)
    _add_gaussians(t, s, start + C_CENTRE * ejection, C_WIDTH * ejection, c_mmhg)
    _add_gaussians(t, s, start + X_CENTRE * ejection, X_WIDTH * ejection, -x_mmhg)
    v = start + V_CENTRE * ejection
    _add_gaussians(t, s, v, V_WIDTH * ejection, v_mmhg)
    _add_gaussians(t, s, v + Y_DELAY_S, np.full(r.size - 1, Y_WIDTH_S), -y_mmhg)
    return s


def cvp(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, duration_s: float,
        mean_mmhg: float, pulse_mmhg: float, c_to_a: float, v_to_a: float, x_to_a: float, y_to_a: float,
        pre_ejection_s: float, p_to_a_s: float, resp_swing_mmhg: float, noise_mmhg: float,
        rng: np.random.Generator) -> np.ndarray:
    """Central venous pressure (mmHg) on the grid `t`. The wave heights are the given ratios to the a wave;
    `pre_ejection_s` is the same draw the ABP uses. The shape is rescaled over 0 <= t < duration_s so its mean
    is `mean_mmhg` and its peak-to-trough range `pulse_mmhg`, then CVP = CVP0 - resp_swing_mmhg x + noise_mmhg eps."""
    s = cvp_shape(t, r_times, heart_rate_bpm, pre_ejection_s, p_to_a_s, 1.0, c_to_a, x_to_a, v_to_a, y_to_a)
    win = (t >= 0.0) & (t < duration_s)
    s = (s - s[win].min()) / (s[win].max() - s[win].min())
    base = mean_mmhg + pulse_mmhg * (s - s[win].mean())
    resp = np.asarray(resp, dtype=np.float64)
    return base - resp_swing_mmhg * resp + noise_mmhg * rng.standard_normal(base.shape)
