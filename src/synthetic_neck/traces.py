"""Ground-truth traces at 1 kHz: ABP, CVP, ECG, finger PPG and respiration, all generated from one cardiac
timeline (R-wave times) and one respiratory waveform, so every delay between them is fixed by construction.

ECG morphology is the Gaussian-sum (closed-form) waveform of the McSharry ECGSYN model:
P. E. McSharry, G. D. Clifford, L. Tarassenko and L. A. Smith, "A dynamical model for generating synthetic
electrocardiogram signals", IEEE Trans Biomed Eng 50(3):289-294, 2003; https://physionet.org/content/ecgsyn/1.0.0/.

Design: docs/superpowers/specs/2026-09-16-physiological-traces-design.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import TraceParams

TRACE_COLUMNS = ("Time", "ABP", "CVP", "ECG", "PPG", "RR")
_CSV_DECIMALS = (4, 3, 3, 4, 4, 4)
CVP_MIN_MMHG, CVP_MAX_MMHG = 2.0, 20.0


def respiration(t: np.ndarray, p: TraceParams) -> np.ndarray:
    """Respiratory phase waveform x(t) in [-1, 1]; +1 is end-inspiration."""
    return np.sin(2 * np.pi * p.resp_rate_bpm / 60.0 * t + p.resp_phase_rad)


def beat_onsets(p: TraceParams, rng: np.random.Generator) -> np.ndarray:
    """R-wave times (s) covering [-2, duration+2] so edge beats are complete. Each RR interval carries Gaussian
    jitter (hr_variability) and respiratory sinus arrhythmia (shorter by rsa_fraction at end-inspiration)."""
    rr = 60.0 / p.heart_rate_bpm
    times = [-2.0]
    while times[-1] < p.duration_s + 2.0:
        x = float(respiration(np.asarray(times[-1]), p))
        times.append(times[-1] + rr * (1.0 + p.hr_variability * rng.standard_normal() - p.rsa_fraction * x))
    return np.asarray(times)


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _gate(phase: np.ndarray, onset: float, width: float = 0.03) -> np.ndarray:
    """Smooth 0 -> 1 switch about `onset`, ~`width` s wide, in place of a hard step."""
    return 0.5 * (1.0 + np.tanh((phase - onset) / width))


def _abp_beat(phase: np.ndarray) -> np.ndarray:
    """Unit-ish central arterial pulse vs time since upstroke: systolic peak, dicrotic wave, run-off."""
    systolic = _gauss(phase, 0.11, 0.045)
    dicrotic = 0.25 * _gauss(phase, 0.33, 0.05)
    runoff = 0.35 * np.exp(-np.clip(phase - 0.30, 0, None) / 0.35) * _gate(phase, 0.30)
    return systolic + dicrotic + runoff


def _ppg_beat(phase: np.ndarray) -> np.ndarray:
    """Peripheral (finger) pulse vs time since its foot: a broad systolic hump peaking ~0.18 s after the foot,
    a dicrotic hump near 0.40 s and a slow run-off."""
    systolic = _gauss(phase, 0.18, 0.075)
    dicrotic = 0.35 * _gauss(phase, 0.40, 0.09)
    runoff = 0.30 * np.exp(-np.clip(phase - 0.35, 0, None) / 0.30) * _gate(phase, 0.35)
    return systolic + dicrotic + runoff


# (name, centre s from R-wave, width s, sign): the CVP beat basis, shared with calibrate.fit_cvp_waves.
CVP_WAVES = (("a", -0.08, 0.045, +1), ("c", 0.06, 0.03, +1), ("x", 0.17, 0.06, -1),
             ("v", 0.33, 0.06, +1), ("y", 0.45, 0.05, -1))
CVP_AMPLITUDE_FIELDS = {"a": "a_wave_mmhg", "c": "c_wave_mmhg", "x": "x_descent_mmhg",
                        "v": "v_wave_mmhg", "y": "y_descent_mmhg"}


def _cvp_beat(phase: np.ndarray, p: TraceParams) -> np.ndarray:
    """CVP waveform (mmHg about zero) vs time since the R-wave: a, c, x, v, y."""
    return sum(sign * getattr(p, CVP_AMPLITUDE_FIELDS[name]) * _gauss(phase, centre, width)
               for name, centre, width, sign in CVP_WAVES)


# (wave, angle deg around the beat, a, b): the ECGSYN defaults of McSharry et al. 2003, in P Q R S T order.
# theta = 0 is the R-peak; a beat is one revolution, so wave positions scale with each beat's RR interval.
ECG_WAVES = (("P", -70.0, 1.2, 0.25), ("Q", -15.0, -5.0, 0.1), ("R", 0.0, 30.0, 0.1),
             ("S", 15.0, -7.5, 0.1), ("T", 100.0, 0.75, 0.4))


def _ecg_beat(theta: np.ndarray) -> np.ndarray:
    """ECGSYN waveform vs angle (rad) around one beat: the exact integral of the model's dz/dt around the limit
    cycle with its baseline-relaxation term dropped, i.e. each wave is a Gaussian of amplitude a * b**2."""
    return sum(a * b ** 2 * np.exp(-0.5 * ((theta - np.deg2rad(deg)) / b) ** 2) for _, deg, a, b in ECG_WAVES)


def generate_trace(p: TraceParams) -> np.ndarray:
    """Return an (N, 6) array with columns Time [s], ABP [mmHg], CVP [mmHg], ECG [mV], PPG [arb], RR [arb].

    ABP is the central pulse delayed to `p.abp_site` (brachial or radial, the latter amplified). The carotid
    pixels are rendered from ABP shifted back by `p.abp_site_delay_s`. RR is a chest-strap style excursion in
    [0, 1], rising on inspiration; ABP and CVP fall on inspiration (spontaneous breathing)."""
    rng = np.random.default_rng(p.seed)
    t = np.arange(p.n_samples) / p.sample_rate_hz
    x = respiration(t, p)
    onsets = beat_onsets(p, rng)

    abp_shape = np.zeros_like(t)
    cvp_shape = np.zeros_like(t)
    ecg_shape = np.zeros_like(t)
    ppg_shape = np.zeros_like(t)
    for r, rr_k in zip(onsets[:-1], np.diff(onsets)):
        abp_shape += _abp_beat(t - (r + p.r_to_abp_foot_s))
        cvp_shape += _cvp_beat(t - r, p)
        ecg_shape += _ecg_beat(2 * np.pi * (t - r) / rr_k)
        ppg_shape += _ppg_beat(t - (r + p.r_to_ppg_foot_s))

    lo, hi = abp_shape.min(), abp_shape.max()
    pulse_pressure = p.systolic_mmhg - p.diastolic_mmhg
    if p.abp_site == "radial":
        pulse_pressure *= p.radial_amplification
    abp = p.diastolic_mmhg + pulse_pressure * (abp_shape - lo) / (hi - lo)
    abp = abp - p.resp_abp_swing_mmhg * x
    abp = abp + p.abp_noise_mmhg * rng.standard_normal(t.shape)

    cvp = p.cvp_mean_mmhg + cvp_shape - p.resp_cvp_swing_mmhg * x
    cvp = cvp + p.cvp_noise_mmhg * rng.standard_normal(t.shape)
    cvp = np.clip(cvp, CVP_MIN_MMHG, CVP_MAX_MMHG)

    ecg = ecg_shape / ecg_shape.max() * p.r_amplitude_mv * (1.0 + p.ecg_r_modulation * x)
    ecg = ecg + p.ecg_wander_mv * x + p.ecg_noise_mv * rng.standard_normal(t.shape)

    lo, hi = ppg_shape.min(), ppg_shape.max()
    ppg = (ppg_shape - lo) / (hi - lo)
    ppg = 0.5 + (ppg - 0.5) * (1.0 + p.ppg_am_frac * x) + p.ppg_wander_frac * x
    ppg = ppg + p.ppg_noise * rng.standard_normal(t.shape)

    rr = 0.5 + 0.5 * x
    return np.column_stack([t, abp, cvp, ecg, ppg, rr])


def write_trace_csv(trace: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_COLUMNS)
        for row in trace:
            w.writerow([f"{value:.{decimals}f}" for value, decimals in zip(row, _CSV_DECIMALS)])


def read_trace_csv(path: Path) -> np.ndarray:
    with Path(path).open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if tuple(header) != TRACE_COLUMNS:
            raise ValueError(f"unexpected trace header {header}")
        return np.asarray([[float(x) for x in row] for row in r])
