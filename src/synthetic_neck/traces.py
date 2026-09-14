"""Ground-truth pressure traces: arterial blood pressure (ABP) and central
venous pressure (CVP), generated from one shared cardiac timeline so the
a-wave precedes the arterial upstroke, the c-wave coincides with it and the
v-wave sits in late systole.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import TraceParams

TRACE_COLUMNS = ("Time", "ABP", "CVP")
CVP_MIN_MMHG, CVP_MAX_MMHG = 2.0, 20.0


def beat_onsets(p: TraceParams, rng: np.random.Generator) -> np.ndarray:
    """R-wave times (s) covering [-2, duration+2] so edge beats are complete."""
    rr = 60.0 / p.heart_rate_bpm
    times = [-2.0]
    while times[-1] < p.duration_s + 2.0:
        times.append(times[-1] + rr * (1.0 + p.hr_variability * rng.standard_normal()))
    return np.asarray(times)


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _abp_beat(phase: np.ndarray) -> np.ndarray:
    """Unit-ish arterial pulse vs time since upstroke: systolic peak, dicrotic wave, run-off."""
    systolic = _gauss(phase, 0.11, 0.045)
    dicrotic = 0.25 * _gauss(phase, 0.33, 0.05)
    runoff = 0.35 * np.exp(-np.clip(phase - 0.30, 0, None) / 0.35) * (phase > 0.30)
    return systolic + dicrotic + runoff


def _cvp_beat(phase: np.ndarray, p: TraceParams) -> np.ndarray:
    """CVP waveform (mmHg about zero) vs time since the R-wave: a, c, x, v, y."""
    a = p.a_wave_mmhg * _gauss(phase, -0.08, 0.045)
    c = p.c_wave_mmhg * _gauss(phase, 0.06, 0.03)
    x = -p.x_descent_mmhg * _gauss(phase, 0.17, 0.06)
    v = p.v_wave_mmhg * _gauss(phase, 0.33, 0.06)
    y = -p.y_descent_mmhg * _gauss(phase, 0.45, 0.05)
    return a + c + x + v + y


def generate_trace(p: TraceParams) -> np.ndarray:
    """Return an (N, 3) array with columns Time [s], ABP [mmHg], CVP [mmHg]."""
    rng = np.random.default_rng(p.seed)
    t = np.arange(p.n_samples) / p.sample_rate_hz
    onsets = beat_onsets(p, rng)

    abp_shape = np.zeros_like(t)
    cvp_shape = np.zeros_like(t)
    for r in onsets:
        abp_shape += _abp_beat(t - (r + p.abp_upstroke_delay_s))
        cvp_shape += _cvp_beat(t - r, p)

    lo, hi = abp_shape.min(), abp_shape.max()
    abp = p.diastolic_mmhg + (p.systolic_mmhg - p.diastolic_mmhg) * (abp_shape - lo) / (hi - lo)

    resp = np.sin(2 * np.pi * p.resp_rate_bpm / 60.0 * t)
    abp = abp + p.resp_abp_swing_mmhg * resp
    cvp = p.cvp_mean_mmhg + cvp_shape - p.resp_cvp_swing_mmhg * resp

    abp = abp + p.abp_noise_mmhg * rng.standard_normal(t.shape)
    cvp = cvp + p.cvp_noise_mmhg * rng.standard_normal(t.shape)
    cvp = np.clip(cvp, CVP_MIN_MMHG, CVP_MAX_MMHG)
    return np.column_stack([t, abp, cvp])


def write_trace_csv(trace: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_COLUMNS)
        for row in trace:
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}"])


def read_trace_csv(path: Path) -> np.ndarray:
    with Path(path).open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if tuple(header) != TRACE_COLUMNS:
            raise ValueError(f"unexpected trace header {header}")
        return np.asarray([[float(x) for x in row] for row in r])
