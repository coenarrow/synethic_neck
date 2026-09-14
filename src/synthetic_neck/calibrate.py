"""Offline calibration: mine the Neckflix dataset for priors.

Reads dataset_info.csv, a seeded sample of trace_data.csv files and a few
frames per sampled recording, and writes aggregated quantiles / histograms
only. No recording IDs, per-patient rows or pixels are written.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import (AppearanceConfig, Choice, GeneratorConfig, GeometryConfig, IlluminationConfig,
                     PulseConfig, Range, SensorConfig, StreamsConfig, TraceConfig, validate)
from .render.camera import gaussian_blur
from .traces import CVP_AMPLITUDE_FIELDS, CVP_WAVES
from .video import FORMATS

PRIORS_SCHEMA_VERSION = 1
QUANTILES = {"p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}
TRACE_TIME, TRACE_CVP, TRACE_ECG = "Time (s)", "CVP (mmHg)", "ECG (mV)"


# --------------------------------------------------------------------------- helpers

def quantiles(values) -> dict[str, float]:
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values to summarise")
    return {k: float(np.percentile(v, p)) for k, p in QUANTILES.items()}


def histogram(values) -> dict[str, float]:
    vals = [str(v) for v in values]
    n = len(vals)
    return {k: vals.count(k) / n for k in sorted(set(vals), key=lambda s: (len(s), s))}


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _range(q: dict, lo_key="p05", hi_key="p95", floor=None, ceil=None) -> Range:
    lo, hi = q[lo_key], q[hi_key]
    if floor is not None:
        lo, hi = max(lo, floor), max(hi, floor)
    if ceil is not None:
        lo, hi = min(lo, ceil), min(hi, ceil)
    return Range(lo, hi)


# --------------------------------------------------------------------------- population

def read_info(root: Path) -> list[dict]:
    with (Path(root) / "dataset_info.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def posture_of(recording_dir: str) -> float:
    return float(recording_dir.split("_")[-2])


def has_depth_ir(recording_dir: str) -> bool:
    return recording_dir.endswith("_D")


def monk_of(row: dict) -> int | None:
    for k in ("Skin_Tone_Clinician", "Skin_Tone_Recorder", "Skin_Tone_Self"):
        v = _num(row.get(k, ""))
        if v is not None and 1 <= v <= 10:
            return int(v)
    return None


def population_priors(rows: list[dict]) -> dict:
    col = lambda k: [x for x in (_num(r.get(k, "")) for r in rows) if x is not None]
    jvp_rows = [r for r in rows if _num(r.get("JVP Height Estimate (cm)", "")) is not None]
    by_posture = {}
    for r in jvp_rows:
        by_posture.setdefault(str(int(posture_of(r["Recording_Directory"]))), []).append(
            float(r["JVP Height Estimate (cm)"]))
    # Pair systolic and diastolic within the same row before differencing, so a row
    # missing one value doesn't get zipped against an unrelated row's value (R14).
    bp_pairs = [(_num(r.get("BP_Sys (mmHG)", "")), _num(r.get("BP_Dia (mmHg)", ""))) for r in rows]
    bp_pairs = [(s, d) for s, d in bp_pairs if s is not None and d is not None]
    dia = [d for _, d in bp_pairs]
    pp = [s - d for s, d in bp_pairs if s > d]
    return {
        "n_rows": len(rows),
        "heart_rate_bpm": quantiles(col("Average_BPM")),
        "age_years": quantiles(col("Age (years)")),
        "neck_circumference_mid_cm": quantiles(col("Neck_Circumference_Mid (cm)")),
        "bp_dia_mmhg": quantiles(dia),
        "pulse_pressure_mmhg": quantiles(pp),
        "sex": histogram(r.get("Sex", "") for r in rows),
        "monk_tone": histogram(m for m in (monk_of(r) for r in rows) if m is not None),
        "posture_deg": histogram(int(posture_of(r["Recording_Directory"])) for r in rows),
        "depth_ir_share": float(np.mean([has_depth_ir(r["Recording_Directory"]) for r in rows])),
        "jvp_height_cm": quantiles(float(r["JVP Height Estimate (cm)"]) for r in jvp_rows),
        "jvp_not_visible_share": 1.0 - len(jvp_rows) / len(rows),
        "posture_jvp_height_cm": {k: quantiles(v) for k, v in sorted(by_posture.items())},
    }


# --------------------------------------------------------------------------- waveform

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, n=len(x))


def detect_r_peaks(ecg: np.ndarray, fs_hz: float, refractory_s: float = 0.3) -> np.ndarray:
    """R-peak sample indices: band-pass 5-30 Hz, square, threshold, refractory."""
    e = _bandpass(ecg, fs_hz, 5.0, 30.0) ** 2
    thr = 0.3 * np.percentile(e, 99)
    above = np.flatnonzero((e[1:-1] > thr) & (e[1:-1] >= e[:-2]) & (e[1:-1] >= e[2:])) + 1
    peaks, last = [], -np.inf
    for i in above:
        if i - last > refractory_s * fs_hz:
            peaks.append(i)
            last = i
        elif e[i] > e[peaks[-1]]:
            peaks[-1], last = i, i
    return np.asarray(peaks, dtype=int)


def _decimate(x: np.ndarray, factor: int) -> np.ndarray:
    n = len(x) // factor * factor
    return x[:n].reshape(-1, factor).mean(1)


def fit_cvp_waves(beat: np.ndarray, tb: np.ndarray, rr_s: float) -> dict[str, float]:
    """Least-squares amplitudes (mmHg, clamped >= 0) of the generator's a/c/x/v/y waves in a beat-averaged
    CVP template. Neighbouring beats (±1, ±2 mean RR) share the amplitudes, so waves that overlap from
    adjacent beats at fast heart rates are modelled instead of misattributed."""
    cols = [sum(sign * np.exp(-0.5 * ((tb - k * rr_s - centre) / width) ** 2) for k in (-2, -1, 0, 1, 2))
            for _, centre, width, sign in CVP_WAVES]
    design = np.column_stack(cols + [np.ones_like(tb)])
    coef, *_ = np.linalg.lstsq(design, beat, rcond=None)
    return {CVP_AMPLITUDE_FIELDS[name]: float(max(coef[i], 0.0)) for i, (name, *_w) in enumerate(CVP_WAVES)}


def waveform_priors_for_recording(trace_csv: Path) -> dict:
    """CVP morphology and rhythm from one 'Time (s),CVP (mmHg),ECG (mV)' file."""
    with open(trace_csv, newline="") as f:
        header = [h.strip() for h in f.readline().split(",")]
    missing = [c for c in (TRACE_TIME, TRACE_CVP, TRACE_ECG) if c not in header]
    if missing:
        raise ValueError(f"{trace_csv} lacks columns {missing}")
    data = np.loadtxt(trace_csv, delimiter=",", skiprows=1, ndmin=2)
    t, cvp, ecg = (data[:, header.index(c)] for c in (TRACE_TIME, TRACE_CVP, TRACE_ECG))
    fs = 1.0 / float(np.median(np.diff(t)))
    peaks = detect_r_peaks(ecg, fs)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 3:
        raise ValueError(f"too few beats detected in {trace_csv}")

    fs_d = 200.0
    factor = max(1, int(round(fs / fs_d)))
    cvp_d, fs_d = _decimate(cvp, factor), fs / factor
    peaks_d = peaks // factor
    pre, post = int(0.3 * fs_d), int(0.7 * fs_d)
    beats = [cvp_d[p - pre:p + post] for p in peaks_d if p - pre >= 0 and p + post <= len(cvp_d)]
    if not beats:
        raise ValueError(f"no complete beat windows in {trace_csv}")
    beat = np.mean(beats, axis=0)
    beat = beat - beat.mean()
    tb = (np.arange(len(beat)) - pre) / fs_d
    waves = fit_cvp_waves(beat, tb, float(rr.mean()))

    resp = _bandpass(cvp_d, fs_d, 0.1, 0.5)
    return {
        "heart_rate_bpm": float(60.0 / rr.mean()),
        "hr_variability": float(rr.std() / rr.mean()),
        "cvp_mean_mmhg": float(cvp.mean()),
        "cvp_pulse_pressure_mmhg": float(np.percentile(cvp_d, 95) - np.percentile(cvp_d, 5)),
        **waves,
        "resp_cvp_swing_mmhg": float(np.sqrt(2) * resp.std()),
        "n_beats": int(len(rr)),
    }


def waveform_priors(per_recording: list[dict]) -> dict:
    keys = [k for k in per_recording[0] if k != "n_beats"]
    out = {k: quantiles(r[k] for r in per_recording) for k in keys}
    out["n_recordings"] = len(per_recording)
    return out


# --------------------------------------------------------------------------- appearance

def read_frames(path: Path, fmt: str, start_s: float, n: int) -> np.ndarray:
    """Decode `n` consecutive frames starting at `start_s` into (n, H, W[, C])."""
    in_fmt, _, dtype, channels = FORMATS[fmt]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start_s}", "-i", str(path), "-frames:v", str(n),
         "-f", "rawvideo", "-pix_fmt", in_fmt, "-"], capture_output=True, check=True).stdout
    shape = (-1, h, w) + ((channels,) if channels > 1 else ())
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def skin_mask(rgb: np.ndarray, central: bool = True) -> np.ndarray:
    """Warm, non-dark, non-blue pixels; optionally restricted to the central half of the frame."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    m = (r > 60) & (r > g) & (g > b) & (r - b > 15)
    if central:
        h, w = m.shape
        box = np.zeros_like(m)
        box[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        m &= box
    return m


def _noise_fit(frames: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Fit temporal variance = read^2 + shot * mean over masked pixels (10 intensity bins)."""
    mean = frames.mean(0)[mask]
    var = frames.var(0)[mask]
    bins = np.linspace(mean.min(), mean.max() + 1e-6, 11)
    idx = np.digitize(mean, bins) - 1
    xs, ys = [], []
    for k in range(10):
        sel = idx == k
        if sel.sum() >= 50:
            xs.append(mean[sel].mean()), ys.append(np.median(var[sel]))
    if len(xs) < 2 or max(xs) - min(xs) < 20.0:
        # Too little intensity spread to separate shot from read noise: treat it all as read noise.
        return float(np.sqrt(np.median(var))), 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(np.sqrt(max(intercept, 0.0))), float(max(slope, 0.0))


def appearance_priors_for_recording(rec_dir: Path, n_frames: int = 10, start_s: float = 10.0) -> dict:
    rgb = read_frames(rec_dir / "K1_RGB.mkv", "rgb", start_s, n_frames).astype(np.float64)
    if rgb.shape[0] < 2:
        start_s = 0.0
        rgb = read_frames(rec_dir / "K1_RGB.mkv", "rgb", start_s, n_frames).astype(np.float64)
    mean_frame = rgb.mean(0)
    mask = skin_mask(mean_frame)
    if mask.sum() < 100:
        raise ValueError(f"too few skin pixels in {rec_dir.name}")
    green = mean_frame[..., 1]
    texture = green - gaussian_blur(green, 3.0)
    read_sd, shot = _noise_fit(rgb[..., 1], mask)
    out = {
        "skin_rgb": [float(mean_frame[..., c][mask].mean()) for c in range(3)],
        "skin_fraction": float(mask.mean()),
        "texture_sd": float(texture[mask].std()),
        "read_noise_sd": read_sd,
        "shot_noise_gain": shot,
    }
    ir_path = rec_dir / "K1_IR.mkv"
    if ir_path.exists():
        ir = read_frames(ir_path, "gray16", start_s, n_frames).astype(np.float64)
        scale = 220.0 / max(np.percentile(ir.mean(0), 99), 1.0)     # map the bright 99th pct to ~220/255
        ir8 = ir * scale
        out["ir_base"] = float(ir8.mean(0)[mask].mean())
        out["ir_read_noise_sd"] = float(np.median(ir8.std(0)[mask]))
    return out


def appearance_priors(per_recording: list[dict], monk_by_index: list[int | None]) -> dict:
    lum = lambda rgb: 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    by_monk: dict[int, list] = {}
    for rec, m in zip(per_recording, monk_by_index):
        if m is not None:
            by_monk.setdefault(m, []).append(rec["skin_rgb"])
    skin_by_monk = {str(m): [float(x) for x in np.mean(v, axis=0)] for m, v in sorted(by_monk.items())}
    ambient = []
    for rec, m in zip(per_recording, monk_by_index):
        if m is not None:
            ambient.append(lum(rec["skin_rgb"]) / lum(skin_by_monk[str(m)]))
    out = {
        "skin_rgb_by_monk": skin_by_monk,
        "skin_fraction": quantiles(r["skin_fraction"] for r in per_recording),
        "texture_sd": quantiles(r["texture_sd"] for r in per_recording),
        "read_noise_sd": quantiles(r["read_noise_sd"] for r in per_recording),
        "shot_noise_gain": quantiles(r["shot_noise_gain"] for r in per_recording),
        "ambient_gain": quantiles(ambient) if ambient else quantiles([1.0]),
    }
    ir = [r for r in per_recording if "ir_base" in r]
    if ir:
        out["ir_base"] = quantiles(r["ir_base"] for r in ir)
        out["ir_read_noise_sd"] = quantiles(r["ir_read_noise_sd"] for r in ir)
    return out


# --------------------------------------------------------------------------- driver

def run_calibration(root: Path, out: Path, n_recordings: int = 30, seed: int = 0) -> dict:
    root, out = Path(root), Path(out)
    rows = read_info(root)
    usable = [r for r in rows if r.get("K1_RGB.mkv") == "TRUE" and r.get("CVP (mmHg)") == "TRUE"
              and r.get("ECG (mV)") == "TRUE" and (root / "data" / r["Recording_Directory"] / "trace_data.csv").exists()]
    if not usable:
        raise FileNotFoundError(f"no usable recordings under {root}")
    rng = np.random.default_rng(seed)
    order = [usable[i] for i in rng.permutation(len(usable))]
    wave, look, monks, skipped = [], [], [], 0
    for r in order:
        if len(wave) == n_recordings:
            break
        d = root / "data" / r["Recording_Directory"]
        try:
            w = waveform_priors_for_recording(d / "trace_data.csv")
            a = appearance_priors_for_recording(d)
        except (ValueError, subprocess.CalledProcessError):
            skipped += 1
            continue
        wave.append(w)
        look.append(a)
        monks.append(monk_of(r))
    if not wave:
        raise ValueError(f"no usable recording under {root} could be calibrated ({skipped} skipped)")
    priors = {
        "schema_version": PRIORS_SCHEMA_VERSION,
        "provenance": {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool_version": __version__,
                       "n_rows": len(rows), "n_recordings": len(wave), "n_skipped": skipped, "seed": seed},
        "population": population_priors(rows),
        "waveform": waveform_priors(wave),
        "appearance": appearance_priors(look, monks),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=2))
    return priors


# --------------------------------------------------------------------------- priors -> config

def config_from_priors(path: Path) -> GeneratorConfig:
    """The `neckflix` preset: p05-p95 of every measured prior mapped onto Ranges."""
    from .presets import MONK_SKIN_RGB   # local import: presets imports this module lazily too

    d = json.loads(Path(path).read_text())
    if d.get("schema_version") != PRIORS_SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {d.get('schema_version')} != {PRIORS_SCHEMA_VERSION}")
    pop, wf, ap = d["population"], d["waveform"], d["appearance"]

    posture = pop["posture_deg"]
    monk = pop["monk_tone"]
    table = [tuple(ap["skin_rgb_by_monk"].get(str(i + 1), MONK_SKIN_RGB[i])) for i in range(10)]

    cfg = GeneratorConfig(
        streams=StreamsConfig(depth_ir_probability=float(pop["depth_ir_share"])),
        trace=TraceConfig(
            heart_rate_bpm=_range(pop["heart_rate_bpm"], floor=40, ceil=140),
            hr_variability=_range(wf["hr_variability"], floor=0.005, ceil=0.15),
            diastolic_mmhg=_range(pop["bp_dia_mmhg"], floor=45, ceil=110),
            pulse_pressure_mmhg=_range(pop["pulse_pressure_mmhg"], floor=25, ceil=90),
            cvp_mean_mmhg=_range(wf["cvp_mean_mmhg"], floor=3, ceil=18),
            a_wave_mmhg=_range(wf["a_wave_mmhg"], floor=0.3),
            c_wave_mmhg=_range(wf["c_wave_mmhg"], floor=0.0),
            x_descent_mmhg=_range(wf["x_descent_mmhg"], floor=0.3),
            v_wave_mmhg=_range(wf["v_wave_mmhg"], floor=0.3),
            y_descent_mmhg=_range(wf["y_descent_mmhg"], floor=0.2),
            resp_cvp_swing_mmhg=_range(wf["resp_cvp_swing_mmhg"], floor=0.2, ceil=4.0),
            posture_deg=Choice(tuple(float(k) for k in posture), tuple(posture.values())),
        ),
        geometry=GeometryConfig(vein_visible_fraction=Range(0.4, 1.0)),
        appearance=AppearanceConfig(
            monk_tone=Choice(tuple(int(k) for k in monk), tuple(monk.values())),
            skin_rgb_by_monk=tuple(table),
            texture_sd=_range(ap["texture_sd"], floor=0.5),
            ir_base=_range(ap["ir_base"], floor=60, ceil=240) if "ir_base" in ap else Range(150, 200),
            shading_strength=Range(0.5, 1.0),
        ),
        pulse=PulseConfig(amplitude_levels=Range(0.5, 2.0), vein_ratio=Range(0.4, 0.8)),
        illumination=IlluminationConfig(
            ambient_gain=_range(ap["ambient_gain"], floor=0.6, ceil=1.4),
            drift_sd=Range(0.0, 0.02), drift_tau_s=Range(10, 60),
            flicker_amp=Range(0.0, 0.005), specular_amp=Range(0.0, 20.0)),
        sensor=SensorConfig(
            read_noise_sd=_range(ap["read_noise_sd"], floor=0.5, ceil=6.0),
            shot_noise_gain=_range(ap["shot_noise_gain"], floor=0.0, ceil=0.1),
            ir_read_noise_sd=_range(ap["ir_read_noise_sd"], floor=0.5, ceil=8.0) if "ir_read_noise_sd" in ap else Range(2, 2),
            blur_sigma_px=Range(0.5, 1.5)),
    )
    validate(cfg)
    return cfg
