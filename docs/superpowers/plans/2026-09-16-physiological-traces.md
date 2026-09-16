# Physiological Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `generate_trace` emits ABP, CVP, ECG, PPG and respiration from one cardiac timeline with literature delays; `--zarr` stores write all five trace groups plus an `abp_site` attr.

**Architecture:** The 1 kHz trace array grows from three to six columns. Every signal hangs off `beat_onsets` (R-wave times, now with respiratory sinus arrhythmia) and one respiratory sinusoid. The stored ABP is delayed to a drawn catheter site; the pulse stage shifts it back by that delay so the carotid pixels keep their timing. The sinks pick columns by name.

**Tech Stack:** numpy only in the generator; zarr 3.3 for the store; matplotlib (the `inspect` extra) for the verification plot.

**Spec:** `docs/superpowers/specs/2026-09-16-physiological-traces-design.md`

## Global Constraints

- No new tests, no pytest runs (user rule). Existing tests that only name a renamed field or unpack three columns get mechanical edits and are listed in the task report; they are never run.
- The only test tool that may be run is remote-physiology's `tools/validate_cache.py`.
- numpy `stack`/`column_stack` idiom, no einops in this package.
- Dependencies only through `uv add`; this plan adds none.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Branch: `feat/physiological-traces` in this repository (already created, holds the spec).
- Every `TraceConfig` default below is copied from spec section 4; do not retune.

---

### Task 1: The six-column trace model

**Files:**
- Modify: `src/synthetic_neck/config.py` (TraceConfig ~L101-118, TraceParams ~L232-256, `_parse_value` ~L463)
- Modify: `src/synthetic_neck/traces.py` (whole module)
- Modify: `tests/test_traces.py` (mechanical edits only, not run)
- Create (throwaway, outside the repo): `<scratchpad>/plot_traces.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `generate_trace(p: TraceParams) -> np.ndarray` of shape `(N, 6)`, columns `TRACE_COLUMNS = ("Time", "ABP", "CVP", "ECG", "PPG", "RR")`; `TraceParams.abp_site: str`, `TraceParams.abp_site_delay_s`, `.r_to_abp_foot_s`, `.r_to_ppg_foot_s` (float properties); `TraceConfig.pep_s` replaces `abp_upstroke_delay_s`; `Choice` overrides accept non-numeric values.

- [ ] **Step 1: Replace the trace block of `TraceConfig`**

In `src/synthetic_neck/config.py`, add `import math` to the imports and replace the `TraceConfig` class with:

```python
@dataclass(frozen=True)
class TraceConfig:
    heart_rate_bpm: Range = Range(55, 95)
    hr_variability: Range = Range(0.01, 0.05)
    resp_rate_bpm: Range = Range(10, 18)
    diastolic_mmhg: Range = Range(62, 88)
    pulse_pressure_mmhg: Range = Range(35, 60)
    cvp_mean_mmhg: Range = Range(6, 15)
    a_wave_mmhg: Range = Range(2.5, 2.5)
    c_wave_mmhg: Range = Range(1.0, 1.0)
    x_descent_mmhg: Range = Range(1.8, 1.8)
    v_wave_mmhg: Range = Range(2.0, 2.0)
    y_descent_mmhg: Range = Range(1.2, 1.2)
    resp_abp_swing_mmhg: Range = Range(2.0, 2.0)
    resp_cvp_swing_mmhg: Range = Range(1.5, 1.5)
    abp_noise_mmhg: Range = Range(0.3, 0.3)
    cvp_noise_mmhg: Range = Range(0.15, 0.15)
    posture_deg: Choice = Choice((0.0, 45.0, 90.0))
    # Timing (s): R-wave -> aortic valve opening, then transit to each measurement site. Sources: spec section 4.
    pep_s: Range = Range(0.08, 0.12)
    abp_site: Choice = Choice(("radial", "brachial"))
    brachial_transit_s: Range = Range(0.05, 0.09)
    radial_transit_s: Range = Range(0.02, 0.04)
    radial_amplification: Range = Range(1.05, 1.15)    # brachial -> radial pulse-pressure amplification
    finger_transit_s: Range = Range(0.12, 0.20)
    # Respiration: phase at t = 0 and the RR shortening at end-inspiration (respiratory sinus arrhythmia).
    resp_phase_rad: Range = Range(0.0, 2 * math.pi)
    rsa_fraction: Range = Range(0.02, 0.08)
    # ECG (mV): McSharry ECGSYN morphology scaled to the R-peak; respiratory modulation; noise.
    r_amplitude_mv: Range = Range(0.8, 1.5)
    ecg_r_modulation: Range = Range(0.03, 0.10)
    ecg_wander_mv: Range = Range(0.02, 0.08)
    ecg_noise_mv: Range = Range(0.005, 0.02)
    # Finger PPG (arb, one beat spans ~0..1): respiratory amplitude modulation, baseline wander, noise.
    ppg_am_frac: Range = Range(0.05, 0.15)
    ppg_wander_frac: Range = Range(0.05, 0.15)
    ppg_noise: Range = Range(0.005, 0.02)
```

`abp_upstroke_delay_s` no longer exists anywhere in the package after this task (`grep -rn abp_upstroke_delay src tests README.md` must return nothing at the end of the task).

- [ ] **Step 2: Replace `TraceParams`**

Replace the `TraceParams` class with:

```python
@dataclass(frozen=True)
class TraceParams:
    duration_s: float = 10.0
    sample_rate_hz: float = 1000.0
    heart_rate_bpm: float = 72.0
    hr_variability: float = 0.03
    resp_rate_bpm: float = 14.0
    systolic_mmhg: float = 120.0
    diastolic_mmhg: float = 78.0
    cvp_mean_mmhg: float = 6.0
    a_wave_mmhg: float = 2.5
    c_wave_mmhg: float = 1.0
    x_descent_mmhg: float = 1.8
    v_wave_mmhg: float = 2.0
    y_descent_mmhg: float = 1.2
    resp_abp_swing_mmhg: float = 2.0
    resp_cvp_swing_mmhg: float = 1.5
    abp_noise_mmhg: float = 0.3
    cvp_noise_mmhg: float = 0.15
    posture_deg: float = 45.0
    pep_s: float = 0.10
    abp_site: str = "brachial"
    brachial_transit_s: float = 0.07
    radial_transit_s: float = 0.03
    radial_amplification: float = 1.10
    finger_transit_s: float = 0.16
    resp_phase_rad: float = 0.0
    rsa_fraction: float = 0.05
    r_amplitude_mv: float = 1.0
    ecg_r_modulation: float = 0.05
    ecg_wander_mv: float = 0.05
    ecg_noise_mv: float = 0.01
    ppg_am_frac: float = 0.10
    ppg_wander_frac: float = 0.10
    ppg_noise: float = 0.01
    seed: int = 0

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sample_rate_hz)) + 1

    @property
    def abp_site_delay_s(self) -> float:
        """Aortic root -> the stored ABP's catheter site."""
        if self.abp_site == "radial":
            return self.brachial_transit_s + self.radial_transit_s
        return self.brachial_transit_s

    @property
    def r_to_abp_foot_s(self) -> float:
        return self.pep_s + self.abp_site_delay_s

    @property
    def r_to_ppg_foot_s(self) -> float:
        """Pulse arrival time: R-wave -> finger PPG foot."""
        return self.pep_s + self.finger_transit_s
```

`sample()` needs no change: `_draw_fields` draws every `TraceConfig` field by name and `TraceParams(**t)` receives them, `abp_site` included (a `Choice` of strings draws a string).

- [ ] **Step 3: Let `Choice` overrides carry strings**

In `_parse_value`, replace the `Choice` branch:

```python
    if hint is Choice:
        return Choice(tuple(_choice_value(p) for p in text.split("|")))
```

and add above `_parse_value`:

```python
def _choice_value(text: str):
    """A Choice value from an override: numeric when it parses as one (posture_deg), else the string (abp_site)."""
    try:
        return float(text)
    except ValueError:
        return text
```

- [ ] **Step 4: Rewrite `traces.py`**

Replace the whole of `src/synthetic_neck/traces.py` with:

```python
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


def _abp_beat(phase: np.ndarray) -> np.ndarray:
    """Unit-ish central arterial pulse vs time since upstroke: systolic peak, dicrotic wave, run-off."""
    systolic = _gauss(phase, 0.11, 0.045)
    dicrotic = 0.25 * _gauss(phase, 0.33, 0.05)
    runoff = 0.35 * np.exp(-np.clip(phase - 0.30, 0, None) / 0.35) * (phase > 0.30)
    return systolic + dicrotic + runoff


def _ppg_beat(phase: np.ndarray) -> np.ndarray:
    """Peripheral (finger) pulse vs time since its foot: a broad systolic hump peaking ~0.18 s after the foot,
    a dicrotic hump near 0.40 s and a slow run-off."""
    systolic = _gauss(phase, 0.18, 0.075)
    dicrotic = 0.35 * _gauss(phase, 0.40, 0.09)
    runoff = 0.30 * np.exp(-np.clip(phase - 0.35, 0, None) / 0.30) * (phase > 0.35)
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
```

Note the beat loop iterates `onsets[:-1]` paired with the following interval; the last onset lies beyond `duration_s + 2` so nothing in `[0, duration_s]` is lost.

- [ ] **Step 5: Mechanical edits to `tests/test_traces.py` (do not run)**

Exactly these four edits, nothing else:

1. `assert tr.shape == (10001, 3)` → `assert tr.shape == (10001, 6)`
2. `t, abp, cvp = tr.T` → `t, abp, cvp = tr.T[:3]`
3. `p = TraceParams(heart_rate_bpm=60, hr_variability=0.0, abp_upstroke_delay_s=0.10)` → `p = TraceParams(heart_rate_bpm=60, hr_variability=0.0, pep_s=0.10)`, and on the next line `t, abp, cvp = generate_trace(p).T` → `t, abp, cvp = generate_trace(p).T[:3]`
4. `== "Time,ABP,CVP"` → `== "Time,ABP,CVP,ECG,PPG,RR"`

- [ ] **Step 6: Eyeball the traces**

Write `<scratchpad>/plot_traces.py` (the scratchpad path is given in the task brief; it is not committed):

```python
"""Throwaway: plot a 5 s window of all six traces for three sampled configs, R-waves marked."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_neck.config import GeneratorConfig, sample
from synthetic_neck.traces import TRACE_COLUMNS, beat_onsets, generate_trace

out = Path(sys.argv[1])
seeds = (1, 2, 3)
fig, axes = plt.subplots(6, len(seeds), figsize=(6 * len(seeds), 13), sharex="col")
for col, seed in enumerate(seeds):
    p = sample(GeneratorConfig(), np.random.default_rng(seed)).trace
    tr = generate_trace(p)
    r = beat_onsets(p, np.random.default_rng(p.seed))
    t = tr[:, 0]
    win = (t >= 2.0) & (t <= 7.0)
    for row, name in enumerate(TRACE_COLUMNS[1:], start=1):
        ax = axes[row - 1, col]
        ax.plot(t[win], tr[win, row], lw=0.8)
        for rk in r[(r >= 2.0) & (r <= 7.0)]:
            ax.axvline(rk, color="k", alpha=0.15, lw=0.6)
        ax.set_ylabel(name)
    axes[0, col].set_title(f"seed {seed}: {p.abp_site}, HR {p.heart_rate_bpm:.0f}, resp {p.resp_rate_bpm:.0f}/min, "
                           f"PAT {p.r_to_ppg_foot_s * 1e3:.0f} ms, R->ABP {p.r_to_abp_foot_s * 1e3:.0f} ms", fontsize=9)
    axes[-1, col].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
```

Run from the package root:

```bash
uv run --extra inspect python <scratchpad>/plot_traces.py <scratchpad>/traces.png
```

Open the PNG (the Read tool renders images) and confirm, per column: the CVP a-wave sits just before each R line; ECG shows P, a narrow QRS on the R line, and a T-wave; the ABP foot is 130 to 250 ms after R; the PPG foot follows the ABP foot and is broader; RR is a clean sinusoid; the spacing of the R lines is visibly tighter where RR is high; ABP and CVP baselines both dip where RR is high. Report what you saw and the PNG path in your report. Fix the code, not the check, if any of these fail.

- [ ] **Step 7: Confirm nothing else names the old field**

```bash
grep -rn "abp_upstroke_delay" src tests README.md docs/superpowers/specs/2026-09-16-physiological-traces-design.md
```

Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add src/synthetic_neck/config.py src/synthetic_neck/traces.py tests/test_traces.py
git commit -m "Generate ECG, finger PPG and respiration beside ABP and CVP

One cardiac timeline with respiratory sinus arrhythmia drives all five
signals. ECG is the McSharry ECGSYN Gaussian sum; ABP is stored at a drawn
radial or brachial site; ABP and CVP now both fall on inspiration.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Carotid timing, five trace groups in the store, README

**Files:**
- Modify: `src/synthetic_neck/render/pulse.py:18-37` (constructor and `pulse_maps`)
- Modify: `src/synthetic_neck/render/renderer.py:27` (PulseStage construction)
- Modify: `src/synthetic_neck/zarr_store.py:1-27` (docstring, `TRACE_COLUMNS`, `TRACE_UNITS`) and the root attrs in `commit` (~L124-133)
- Modify: `src/synthetic_neck/generate.py` (`derived` dict in `generate_sample`, ~L85-94)
- Modify: `README.md:28-42`

**Interfaces:**
- Consumes: `TraceParams.abp_site`, `.abp_site_delay_s`, `.r_to_abp_foot_s`, `.r_to_ppg_foot_s` and the six-column trace from Task 1.
- Produces: stores with `abp`, `cvp`, `ecg`, `ppg`, `rr` under every modality and root attr `abp_site`.

- [ ] **Step 1: Shift the carotid drive**

In `src/synthetic_neck/render/pulse.py`, change the constructor signature and body, and `pulse_maps`:

```python
class PulseStage:
    def __init__(self, pulse: PulseParams, gp: GeometryParams, geometry: VesselGeometry,
                 pixel_scale_mm: float, trace: np.ndarray, site_delay_s: float = 0.0):
        self.p = pulse
        self.t = trace[:, 0]
        self.abp_n = normalise(trace[:, 1])
        self.cvp_n = normalise(trace[:, 2])
        # The stored ABP is measured at an arm site `site_delay_s` after the aortic root; the carotid pulse is
        # the same waveform that much earlier, before its own root -> neck propagation (delay_art).
        self.site_delay_s = float(site_delay_s)
        self.map_art = float(trace[:, 1].mean())
        ...  # rest unchanged

    def pulse_maps(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised pressure at every pixel of each vessel at `time_s`."""
        p_art = np.interp(time_s - self.delay_art + self.site_delay_s, self.t, self.abp_n)
        p_vein = np.interp(time_s - self.delay_vein, self.t, self.cvp_n)
        return p_art, p_vein
```

Keep every other line of the constructor as it is. In `src/synthetic_neck/render/renderer.py` change the construction to:

```python
        self.pulse = PulseStage(params.pulse, params.geometry, g, self.pixel_scale_mm, trace,
                                site_delay_s=params.trace.abp_site_delay_s)
```

The default `site_delay_s=0.0` keeps `tests/test_render_pulse.py`'s direct construction importable; do not edit that file.

- [ ] **Step 2: Five trace groups and `abp_site` in the store**

In `src/synthetic_neck/zarr_store.py` replace the two constants:

```python
TRACE_COLUMNS = {"abp": 1, "cvp": 2, "ecg": 3, "ppg": 4, "rr": 5}          # columns of generate_trace's array
TRACE_UNITS = {"abp": "mmHg", "cvp": "mmHg", "ecg": "mV", "ppg": "arb", "rr": "arb"}
```

Update the module docstring's layout line to:

```text
        `-- rgb | ir | depth  video/data (C, T, H, W); timestamps_us/data (T,) int64;
                              abp, cvp (mmHg), ecg (mV), ppg, rr (arb): <trace>/data (T,) float64, attrs units
```

and its root-attrs line to include `abp_site`. In `commit`, add one entry to `attrs` after `"posture"`:

```python
            "abp_site": params["trace"]["abp_site"],
```

- [ ] **Step 3: Derived delays in the metadata**

In `generate_sample` (`src/synthetic_neck/generate.py`), add to the `"derived"` dict after `"vein_visible_fraction"`:

```python
                "abp_site": params.trace.abp_site,
                "abp_site_delay_s": params.trace.abp_site_delay_s,
                "r_to_abp_foot_s": params.trace.r_to_abp_foot_s,
                "r_to_ppg_foot_s": params.trace.r_to_ppg_foot_s,
```

- [ ] **Step 4: README**

In `README.md`, replace the sentence fragment `` `trace.csv` (Time, ABP, CVP at 1 kHz), `` with `` `trace.csv` (Time, ABP, CVP, ECG, PPG, RR at 1 kHz; see Traces), ``. In the zarr section replace `Each modality carries per-frame timestamps and ABP/CVP in mmHg interpolated to the frames.` with `Each modality carries per-frame timestamps and the five traces (` `abp`, `cvp` in mmHg, `ecg` in mV, `ppg` and `rr` in arb) interpolated to the frames.` and add `abp_site` to the listed root attrs (`participant` (the sample index), `posture`, `abp_site`, ...).

Then add a section between "Generate" and "Inspect":

```markdown
## Traces

Every sample carries five ground-truth traces at 1 kHz, generated from one
cardiac timeline (R-wave times with respiratory sinus arrhythmia) and one
respiratory sinusoid, so the delays between them are fixed by construction:

| Column | Units | Model |
| --- | --- | --- |
| `ABP` | mmHg | central pulse (systolic peak, dicrotic wave, run-off) delayed to a drawn catheter site, `radial` or `brachial`, recorded as `trace.abp_site`; radial pulse pressure is amplified |
| `CVP` | mmHg | a, c, x, v, y waves anchored on the R-wave |
| `ECG` | mV | the McSharry ECGSYN model (McSharry, Clifford, Tarassenko, Smith, *IEEE Trans Biomed Eng* 50(3):289–294, 2003) as its closed-form Gaussian sum, scaled to the drawn R amplitude |
| `PPG` | arb | finger pulse: broad systolic hump, dicrotic hump, run-off; foot at R + `pep_s` + `finger_transit_s` |
| `RR` | arb | chest excursion in [0, 1], rising on inspiration |

Respiration modulates everything: ABP and CVP fall on inspiration, ECG
gets baseline wander and R-amplitude modulation, PPG gets baseline wander and
amplitude modulation, and RR intervals shorten on inspiration. The timing
fields and their literature sources are tabulated in
`docs/superpowers/specs/2026-09-16-physiological-traces-design.md`. The
carotid pixels are rendered from ABP shifted back to central timing.
```

- [ ] **Step 5: Validator run**

From the remote-physiology root (`/Users/20759193/repos/remote-physiology`), with `<scratch>` the scratchpad directory from the brief:

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset lesson --n 4 --set video.frame_size=64 \
    --set streams.depth_ir_probability=0.5 --out <scratch>/synthetic_zarr
uv run python tools/validate_cache.py <scratch>/synthetic_zarr
```

Expected: `4/4 stores pass`. Then confirm the new groups and attr:

```bash
uv run python -c "
import zarr, sys
g = zarr.open_group('<scratch>/synthetic_zarr/0.zarr', mode='r')
print(g.attrs['abp_site'], sorted(g['1/rgb'].group_keys()))
print({k: g['1/rgb'][k].attrs['units'] for k in ('abp','cvp','ecg','ppg','rr')})
print(g['1/rgb/ecg/data'][:5], g['1/rgb/rr/data'][:5])"
```

Expected: `radial` or `brachial`; the group list includes `abp cvp ecg ppg rr timestamps_us video`; the units dict matches `TRACE_UNITS`; the ECG and RR values are finite floats.

- [ ] **Step 6: Folder-mode run**

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate \
    --preset lesson --n 1 --set video.frame_size=64 --out <scratch>/synthetic_folder
ls <scratch>/synthetic_folder/0
head -2 <scratch>/synthetic_folder/0/trace.csv
```

Expected: the five files `trace.csv gray_video.mkv rgbid_video.mkv vessel_ids.npy metadata.json`, and the header `Time,ABP,CVP,ECG,PPG,RR`. (Requires ffmpeg on PATH; if it is missing, say so in the report instead of skipping silently.)

- [ ] **Step 7: Commit**

```bash
git add src/synthetic_neck/render/pulse.py src/synthetic_neck/render/renderer.py \
        src/synthetic_neck/zarr_store.py src/synthetic_neck/generate.py README.md
git commit -m "Store ecg, ppg and rr traces and abp_site; keep the carotid on central timing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: remote-physiology cache spec and dataset config

**Files (all in `/Users/20759193/repos/remote-physiology`, branch `feat/synthetic-neck-traces` off `main`):**
- Modify: `dataset/data_loader/SYNTHETIC_NECK.md` (layout block ~L42-50, trace bullets ~L61-63, root attrs ~L78-90)
- Modify: `configs/datasets/synthetic_neck.yaml` (header comment, after the `posture` line)

**Interfaces:**
- Consumes: the store layout produced by Task 2.
- Produces: docs only.

- [ ] **Step 1: Branch**

```bash
git -C /Users/20759193/repos/remote-physiology checkout -b feat/synthetic-neck-traces main
```

- [ ] **Step 2: Cache spec**

In `dataset/data_loader/SYNTHETIC_NECK.md`:

1. In the opening sentence, replace `with ground-truth arterial (ABP) and central venous (CVP) pressure` with `with ground-truth arterial (ABP) and central venous (CVP) pressure, ECG, finger PPG and respiration`.
2. In the layout tree, after the `cvp/data` line add three lines with the same indentation:

   ```text
   |   |-- ecg/data            (T,) float64        attrs: units="mV"
   |   |-- ppg/data            (T,) float64        attrs: units="arb"
   |   `-- rr/data             (T,) float64        attrs: units="arb"
   ```

   and change the `cvp/data` line's tree glyph from `` ` `` to `|` so the tree stays well-formed.
3. Replace the `**abp, cvp**` bullet with:

   ```markdown
   - **`abp`, `cvp`, `ecg`, `ppg`, `rr`**: the generator's 1 kHz traces, linearly
     interpolated at the frame times, identical under every modality. ABP is
     stored at a drawn catheter site (`abp_site`, radial or brachial); the
     carotid pixels are rendered from it shifted back to central timing. ECG is
     the McSharry ECGSYN waveform in mV. PPG is a finger pulse and `rr` a chest
     excursion in [0, 1], both `arb`. All five share one cardiac timeline and
     one respiratory waveform; the generator's spec
     (`docs/superpowers/specs/2026-09-16-physiological-traces-design.md` in the
     submodule) tabulates the delays and their sources.
   ```
4. In the root-attributes block, after the `"posture"` line add:

   ```text
     "abp_site": "radial",        # radial | brachial, from trace.abp_site
   ```

- [ ] **Step 3: Dataset YAML header**

In `configs/datasets/synthetic_neck.yaml`, after the `posture` header line add:

```yaml
#   abp_site           radial | brachial; the stored ABP's catheter site
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/20759193/repos/remote-physiology add dataset/data_loader/SYNTHETIC_NECK.md configs/datasets/synthetic_neck.yaml
git -C /Users/20759193/repos/remote-physiology commit -m "docs: synthetic-neck stores carry ecg, ppg and rr traces and abp_site

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

The submodule pointer bump is done at finish time, after the submodule branch is merged, not in this task.
