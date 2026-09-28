"""ECG: R-wave times from a walk driven by the respiratory waveform (RSA), jitter and a Mayer wave, then the
ECGSYN beat morphology of McSharry et al. 2003 placed on those times."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import Range

MAYER_HZ = 0.1

# (wave, centre deg, amplitude a, width b rad): the ECGSYN defaults of McSharry et al. 2003, P Q R S T.
ECG_WAVES = (("P", -70.0, 1.2, 0.25),
             ("Q", -15.0, -5.0, 0.1),
             ("R", 0.0, 30.0, 0.1),
             ("S", 15.0, -7.5, 0.1),
             ("T", 100.0, 0.75, 0.4))


@dataclass(frozen=True)
class EcgConfig:
    heart_rate_bpm: Range
    hr_variability: Range
    rsa_fraction: Range
    mayer_fraction: Range
    r_amplitude: Range
    am_fraction: Range
    wander: Range
    noise: Range


def r_wave_times(t: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, hr_variability: float,
                 rsa_fraction: float, mayer_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """R-wave times (s) spanning the grid `t`. A walk: starting at a random point within the first mean RR,
    each interval is the mean RR times (1 + jitter - rsa_fraction * x + mayer), where jitter is Gaussian with
    sd `hr_variability`, x is `resp` at the previous R wave, and mayer is a 0.1 Hz sinusoid of amplitude
    `mayer_fraction` with a random phase. Intervals are floored at 0.3 mean RR."""
    t = np.asarray(t, dtype=np.float64)
    rr0 = 60.0 / heart_rate_bpm
    phase = rng.uniform(0.0, 2 * np.pi)
    times = [t[0] + rng.uniform(0.0, rr0)]
    while times[-1] < t[-1]:
        t_k = times[-1]
        x = float(np.interp(t_k, t, resp))
        step = 1.0 + hr_variability * rng.standard_normal() - rsa_fraction * x \
            + mayer_fraction * np.sin(2 * np.pi * MAYER_HZ * t_k + phase)
        times.append(t_k + rr0 * max(step, 0.3))
    return np.asarray(times)


def ecg_beat(theta: np.ndarray, heart_rate_bpm: float = 60.0) -> np.ndarray:
    """One ECG beat vs angle (rad) around the cycle, theta = 0 at the R peak: the McSharry Gaussian sum,
    z = sum_i a_i b_i^2 exp(-wrap(theta - theta_i)^2 / 2 b_i^2), the closed-form limit cycle of ECGSYN's z
    equation. As in ECGSYN, widths and wave positions grow in angle with heart rate: eta = sqrt(hr / 60)
    scales every b_i, the Q and S centres, and the P and T centres by sqrt(eta). Since a beat is a shorter
    time at high rates, the waves move earlier in time but less than proportionally (roughly Bazett)."""
    theta = np.asarray(theta, dtype=np.float64)
    eta = np.sqrt(heart_rate_bpm / 60.0)
    centre_scale = {"P": np.sqrt(eta), "Q": eta, "R": 1.0, "S": eta, "T": np.sqrt(eta)}
    z = np.zeros_like(theta)
    for name, deg, a, b in ECG_WAVES:
        d = theta - np.deg2rad(deg) * centre_scale[name]
        d = d - 2 * np.pi * np.round(d / (2 * np.pi))          # wrap to (-pi, pi]
        b = b * eta
        z += a * b * b * np.exp(-0.5 * (d / b) ** 2)
    return z


def beat_angle(t: np.ndarray, r_times: np.ndarray) -> np.ndarray:
    """Angle (rad) of each sample within the beat containing it: 0 at r_k, 2 pi at r_{k+1}."""
    t, r = np.asarray(t, dtype=np.float64), np.asarray(r_times, dtype=np.float64)
    k = np.clip(np.searchsorted(r, t, side="right") - 1, 0, r.size - 2)
    return 2 * np.pi * (t - r[k]) / (r[k + 1] - r[k])


def t_wave_end(r_times: np.ndarray, heart_rate_bpm: float) -> np.ndarray:
    """End of the T wave (s) of each beat but the last: the T Gaussian's centre plus two widths, with the ECGSYN
    heart-rate scaling of `ecg_beat`, converted to time with that beat's RR interval. Length len(r_times) - 1.
    This is the aortic valve closure that the ABP reads."""
    r = np.asarray(r_times, dtype=np.float64)
    eta = np.sqrt(heart_rate_bpm / 60.0)
    _, deg, _, b = next(w for w in ECG_WAVES if w[0] == "T")
    theta_end = np.deg2rad(deg) * np.sqrt(eta) + 2.0 * b * eta
    return r[:-1] + theta_end / (2 * np.pi) * np.diff(r)


def p_wave_time(r_times: np.ndarray, heart_rate_bpm: float) -> np.ndarray:
    """Centre of the P wave (s) preceding each beat but the first: the P Gaussian's centre with the ECGSYN
    heart-rate scaling of `ecg_beat`, converted to time with the RR interval it lies in. Length len(r_times) - 1,
    element k being the P wave before r_times[k + 1]. This is the atrial contraction that the CVP reads."""
    r = np.asarray(r_times, dtype=np.float64)
    eta = np.sqrt(heart_rate_bpm / 60.0)
    _, deg, _, _ = next(w for w in ECG_WAVES if w[0] == "P")
    theta_p = np.deg2rad(deg) * np.sqrt(eta)                    # negative: before the R peak
    return r[1:] + theta_p / (2 * np.pi) * np.diff(r)


def ecg(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, r_amplitude: float,
        am_fraction: float, wander: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """ECG (arbitrary units) on the grid `t`: the beat shape on `r_times`, scaled so the R peak is `r_amplitude`, then
    ECG = ECG0 (1 + am_fraction x) + wander x + noise eps, with x = `resp`."""
    z = ecg_beat(beat_angle(t, r_times), heart_rate_bpm)
    base = r_amplitude * z / z.max()
    resp = np.asarray(resp, dtype=np.float64)
    return base * (1.0 + am_fraction * resp) + wander * resp + noise * rng.standard_normal(base.shape)
