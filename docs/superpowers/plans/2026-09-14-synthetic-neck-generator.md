# Synthetic Neck Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `synthetic-neck` CLI that generates ground-truth-labelled synthetic neck videos (RGB, IR, depth) with arterial and jugular pulsation, configurable through three presets (`lesson`, `benchmark`, `neckflix`) plus dotted-name overrides, with a separate offline `calibrate` command that mines the real Neckflix dataset for priors.

**Architecture:** One `GeneratorConfig` dataclass of `Range`/`Choice` fields is sampled into concrete `SampleParams`; a renderer built from four stages (base scene → pulse → illumination → sensor) turns params plus a synthesised ABP/CVP trace into frames; ffmpeg pipes write lossless FFV1 MKV. Presets are functions returning a `GeneratorConfig`; the `neckflix` preset reads `priors/neckflix.json`.

**Tech Stack:** Python 3.13, uv, numpy only at runtime (calibration included, stdlib csv), ffmpeg/ffprobe on PATH, pytest. Optional `inspect` extra: matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-14-synthetic-neck-generator-design.md`

## Global Constraints

- `requires-python = ">=3.13"`, project managed with `uv`; run everything as `uv run ...`.
- Runtime dependency list is exactly `["numpy>=2.0"]`. `inspect` extra: `["matplotlib>=3.9"]`. Dev group: `["pytest>=8"]`. The spec's optional pandas/h5py extra is not needed: calibration reads CSV with the stdlib and never opens `EV.hdf5`.
- Package name `synthetic_neck` under `src/`; CLI entry point `synthetic-neck`.
- Every sample must contain a visible arterial and venous pulse; `geometry.vein_visible_fraction.lo` must be `>= 0.1` (validated) and presets use `lesson` 1.0–1.0, `benchmark` 0.6–1.0, `neckflix` 0.4–1.0.
- No real Neckflix pixels or traces are ever written into generated samples or into `priors/neckflix.json`; the priors file holds only quantiles/histograms and a provenance block.
- Output layout per sample is unchanged: `trace.csv`, `gray_video.mkv`, `rgbid_video.mkv`, `vessel_ids.npy`, `metadata.json`.
- Artery pulse amplitude targets (8-bit levels): `lesson` 7–12, `benchmark` 2–5, `neckflix` 0.5–2.
- Neckflix facts: `_D/_N` suffix = depth+IR recorded / not; skin tone columns are Monk Skin Tone (1–10); `0/45/90` in folder names is participant posture; `Not Visible` JVP rows are excluded from JVP priors.
- Commit after every task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Tests that need ffmpeg are marked with `needs_ffmpeg` (skip if absent). Tests never touch the Neckflix drive.

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, hatch build, `synthetic-neck` script, extras |
| `src/synthetic_neck/__init__.py` | version string |
| `src/synthetic_neck/config.py` | `Range`, `IntRange`, `Choice`, config blocks, `GeneratorConfig`, `SampleParams`, `sample()`, `validate()`, `apply_override()`, JSON round trip |
| `src/synthetic_neck/presets.py` | `lesson()`, `benchmark()`, `neckflix()`, `PRESETS`, Monk skin table |
| `src/synthetic_neck/traces.py` | `generate_trace(TraceParams)`, CSV IO |
| `src/synthetic_neck/geometry.py` | `VesselGeometry`, `CameraParams`, `vessel_geometry()`, `tube_fields`, `visibility_taper`, `id_map`, PWV, `delay_map`, `scaled` |
| `src/synthetic_neck/render/base.py` | `BaseScene` |
| `src/synthetic_neck/render/pulse.py` | `PulseStage` |
| `src/synthetic_neck/render/illumination.py` | `IlluminationStage` |
| `src/synthetic_neck/render/camera.py` | `SensorStage`, `gaussian_blur`, `area_resample_matrix`, `depth_to_uint16`, `to_gray` |
| `src/synthetic_neck/render/renderer.py` | `Renderer` composing the stages |
| `src/synthetic_neck/video.py` | `MkvWriter`, `mux`, `read_mkv` |
| `src/synthetic_neck/generate.py` | `generate_sample`, `generate_dataset` |
| `src/synthetic_neck/inspect.py` | FFT QA maps |
| `src/synthetic_neck/calibrate.py` | Neckflix → `priors/neckflix.json` |
| `src/synthetic_neck/cli.py` | argparse front end |
| `priors/neckflix.json` | checked-in calibration output |
| `tests/conftest.py`, `tests/test_*.py` | one test module per source module |

---

### Task 1: Project scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `src/synthetic_neck/__init__.py`, `tests/conftest.py`, `tests/test_package.py`
- Delete: `main.py`

**Interfaces:**
- Produces: importable package `synthetic_neck` with `__version__ = "0.1.0"`; the `needs_ffmpeg` skip marker defined in `tests/conftest.py`, which test modules import with `from conftest import needs_ffmpeg`.

- [ ] **Step 1: Replace pyproject.toml**

```toml
[project]
name = "synthetic-neck"
version = "0.1.0"
description = "Synthetic neck videos with ground-truth arterial and jugular pulsation"
readme = "README.md"
requires-python = ">=3.13"
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
inspect = ["matplotlib>=3.9"]

[project.scripts]
synthetic-neck = "synthetic_neck.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/synthetic_neck"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the package and conftest**

`src/synthetic_neck/__init__.py`:
```python
"""Synthetic neck video generator."""
__version__ = "0.1.0"
```

`tests/conftest.py`:
```python
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets tests do `from conftest import needs_ffmpeg`

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
```

`tests/test_package.py`:
```python
import synthetic_neck


def test_version():
    assert synthetic_neck.__version__ == "0.1.0"
```

- [ ] **Step 3: Remove the placeholder, sync, run**

```bash
rm main.py
uv sync --all-extras
uv run pytest -q
```
Expected: `1 passed`.

- [ ] **Step 4: Commit (folds in the untracked scaffold files)**

```bash
git add pyproject.toml uv.lock .python-version .gitignore README.md src tests
git commit -m "Scaffold synthetic_neck package

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Lossless video IO (`video.py`)

**Files:**
- Create: `src/synthetic_neck/video.py`, `tests/test_video.py`

**Interfaces:**
- Produces: `MkvWriter(path, size=(h, w), fmt="rgb"|"gray"|"gray16", fps)` context manager with `.write(frame)`; `mux(inputs: list[Path], out: Path, titles: list[str] | None)`; `read_mkv(path, fmt, stream=0) -> np.ndarray (T,H,W[,C])`; `FORMATS`.

- [ ] **Step 1: Write the failing test**

`tests/test_video.py`:
```python
import numpy as np
from conftest import needs_ffmpeg

from synthetic_neck.video import MkvWriter, mux, read_mkv


@needs_ffmpeg
def test_mkv_round_trip_is_lossless(tmp_path):
    rng = np.random.default_rng(1)
    for fmt, shape, dtype, hi in (("gray", (5, 32, 48), np.uint8, 256), ("rgb", (5, 32, 48, 3), np.uint8, 256),
                                  ("gray16", (5, 32, 48), np.uint16, 65536)):
        frames = rng.integers(0, hi, size=shape, dtype=dtype)
        with MkvWriter(tmp_path / f"{fmt}.mkv", (32, 48), fmt) as w:
            for f in frames:
                w.write(f)
        np.testing.assert_array_equal(read_mkv(tmp_path / f"{fmt}.mkv", fmt), frames)
    mux([tmp_path / "rgb.mkv", tmp_path / "gray.mkv", tmp_path / "gray16.mkv"], tmp_path / "mux.mkv",
        titles=["rgb", "ir", "depth"])
    assert read_mkv(tmp_path / "mux.mkv", "gray16", stream=2).shape == (5, 32, 48)


@needs_ffmpeg
def test_writer_rejects_wrong_shape(tmp_path):
    import pytest
    with MkvWriter(tmp_path / "x.mkv", (4, 4), "gray") as w:
        with pytest.raises(ValueError):
            w.write(np.zeros((4, 5), np.uint8))
        w.write(np.zeros((4, 4), np.uint8))
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: synthetic_neck.video`.

- [ ] **Step 3: Implement video.py**

```python
"""Lossless MKV writing via an ffmpeg subprocess (FFV1 codec, VLC-playable)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

# name -> (raw pix_fmt fed to ffmpeg, FFV1 storage pix_fmt, numpy dtype, channels)
FORMATS = {
    "gray": ("gray", "gray", np.uint8, 1),
    "rgb": ("rgb24", "bgr0", np.uint8, 3),      # bgr0 = lossless packed RGB
    "gray16": ("gray16le", "gray16le", np.uint16, 1),
}


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH")


class MkvWriter:
    """Write one video stream of `fmt` frames to an FFV1/MKV file."""

    def __init__(self, path: Path, size: tuple[int, int], fmt: str = "rgb", fps: float = 30.0):
        require_ffmpeg()
        in_fmt, out_fmt, self.dtype, self.channels = FORMATS[fmt]
        self.path, self.size = Path(path), size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = size
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "ffv1", "-level", "3", "-pix_fmt", out_fmt,
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        expected = self.size + ((self.channels,) if self.channels > 1 else ())
        if frame.shape != expected or frame.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype.__name__} {expected}, got {frame.dtype} {frame.shape}")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {err}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def mux(inputs: list[Path], out: Path, titles: list[str] | None = None) -> None:
    """Combine single-stream MKVs into one multi-stream MKV (stream copy)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    for i in range(len(inputs)):
        cmd += ["-map", str(i)]
    for i, t in enumerate(titles or []):
        cmd += [f"-metadata:s:v:{i}", f"title={t}"]
    cmd += ["-c", "copy", str(out)]
    subprocess.run(cmd, check=True)


def read_mkv(path: Path, fmt: str = "rgb", stream: int = 0) -> np.ndarray:
    """Decode one video stream of an MKV into a (T, H, W[, C]) array."""
    in_fmt, _, dtype, channels = FORMATS[fmt]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"v:{stream}", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:v:{stream}",
         "-f", "rawvideo", "-pix_fmt", in_fmt, "-"],
        capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=dtype)
    shape = (-1, h, w) + ((channels,) if channels > 1 else ())
    return arr.reshape(shape)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_video.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/video.py tests/test_video.py
git commit -m "Add lossless MKV writer/reader via ffmpeg

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Configuration model (`config.py`)

**Files:**
- Create: `src/synthetic_neck/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Range(lo, hi).draw(rng) -> float`, `IntRange(lo, hi).draw(rng) -> int`, `Choice(values, weights=None).draw(rng)`.
  - Config blocks: `VideoConfig`, `StreamsConfig`, `TraceConfig`, `CameraConfig`, `GeometryConfig`, `AppearanceConfig`, `PulseConfig`, `IlluminationConfig`, `SensorConfig`, `MotionConfig`, `RhythmConfig`; container `GeneratorConfig`.
  - Concrete params: `VideoParams`, `StreamsParams`, `TraceParams`, `CameraParams`, `GeometryParams`, `AppearanceParams`, `PulseParams`, `IlluminationParams`, `SensorParams`; container `SampleParams`.
  - `sample(config: GeneratorConfig, rng: np.random.Generator) -> SampleParams`
  - `validate(config: GeneratorConfig) -> None` (raises `ConfigError`)
  - `apply_override(config, key: str, text: str) -> GeneratorConfig`
  - `config_to_dict(config) -> dict`, `config_from_dict(d) -> GeneratorConfig`, `params_to_dict(params) -> dict`
  - `vein_visible_fraction(posture_deg, cvp_mean_mmhg, length_mm, bounds: Range) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
import json

import numpy as np
import pytest

from synthetic_neck.config import (
    Choice, ConfigError, GeneratorConfig, IntRange, Range, apply_override, config_from_dict,
    config_to_dict, params_to_dict, sample, validate, vein_visible_fraction,
)


def test_range_draws_within_bounds_and_fixed_when_degenerate():
    rng = np.random.default_rng(0)
    r = Range(2.0, 3.0)
    xs = [r.draw(rng) for _ in range(100)]
    assert all(2.0 <= x <= 3.0 for x in xs) and min(xs) < 2.2 and max(xs) > 2.8
    assert Range(5.0, 5.0).draw(rng) == 5.0
    assert IntRange(1, 3).draw(rng) in (1, 2, 3)
    assert Choice((0, 45, 90), (0.0, 0.0, 1.0)).draw(rng) == 90
    with pytest.raises(ConfigError):
        Range(3.0, 2.0)


def test_default_config_validates_and_samples():
    cfg = GeneratorConfig()
    validate(cfg)
    p = sample(cfg, np.random.default_rng(1))
    assert p.video.frame_size == 300 and p.trace.duration_s == 10.0
    assert 55 <= p.trace.heart_rate_bpm <= 95
    assert p.trace.systolic_mmhg > p.trace.diastolic_mmhg
    assert p.geometry.artery_side in (-1, 1)
    assert 0.4 <= p.geometry.centre_frac_xy[0] <= 0.6
    assert p.streams.has_depth_ir is True
    assert len(p.appearance.skin_rgb) == 3


def test_sampling_is_deterministic():
    cfg = GeneratorConfig()
    a = params_to_dict(sample(cfg, np.random.default_rng(9)))
    b = params_to_dict(sample(cfg, np.random.default_rng(9)))
    assert a == b


def test_override_parsing():
    cfg = GeneratorConfig()
    cfg = apply_override(cfg, "pulse.amplitude_levels", "0.5,2")
    assert cfg.pulse.amplitude_levels == Range(0.5, 2.0)
    cfg = apply_override(cfg, "pulse.amplitude_levels", "4")
    assert cfg.pulse.amplitude_levels == Range(4.0, 4.0)
    cfg = apply_override(cfg, "video.frame_size", "64")
    assert cfg.video.frame_size == 64
    cfg = apply_override(cfg, "trace.posture_deg", "0|45")
    assert cfg.trace.posture_deg.values == (0.0, 45.0)
    cfg = apply_override(cfg, "streams.depth_ir_probability", "0.5")
    assert cfg.streams.depth_ir_probability == 0.5
    with pytest.raises(ConfigError, match="unknown"):
        apply_override(cfg, "pulse.nope", "1")
    with pytest.raises(ConfigError, match="unknown"):
        apply_override(cfg, "nope.x", "1")
    with pytest.raises(ConfigError):
        apply_override(cfg, "pulse.amplitude_levels", "abc")


def test_validation_rejects_bad_configs():
    cfg = apply_override(GeneratorConfig(), "geometry.vein_visible_fraction", "0.05,1")
    with pytest.raises(ConfigError, match="vein_visible_fraction"):
        validate(cfg)
    cfg = apply_override(GeneratorConfig(), "geometry.length_mm", "400,400")
    with pytest.raises(ConfigError, match="fit"):
        validate(cfg)


def test_json_round_trip():
    cfg = apply_override(GeneratorConfig(), "illumination.flicker_amp", "0.01,0.02")
    d = config_to_dict(cfg)
    assert json.loads(json.dumps(d)) == d
    assert config_from_dict(d) == cfg


def test_vein_visible_fraction():
    b = Range(0.4, 1.0)
    assert vein_visible_fraction(0, 6.0, 60.0, b) == 1.0          # supine: whole column visible
    upright = vein_visible_fraction(90, 6.0, 60.0, b)
    mid = vein_visible_fraction(45, 6.0, 60.0, b)
    assert 0.4 <= upright <= mid <= 1.0
    assert vein_visible_fraction(90, 15.0, 60.0, b) > upright     # higher CVP -> taller column
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement config.py**

```python
"""Configuration (ranges) -> concrete per-sample parameters.

A `GeneratorConfig` is a tree of blocks whose random fields are `Range`,
`IntRange` or `Choice`. `sample()` draws each of them once, in a fixed order,
producing `SampleParams`: parallel dataclasses of plain numbers that the
trace synthesiser and renderer consume.
"""
from __future__ import annotations

import types
import typing
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- distributions

@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def __post_init__(self):
        object.__setattr__(self, "lo", float(self.lo))
        object.__setattr__(self, "hi", float(self.hi))
        if self.lo > self.hi:
            raise ConfigError(f"Range lo {self.lo} > hi {self.hi}")

    def draw(self, rng: np.random.Generator) -> float:
        return self.lo if self.lo == self.hi else float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class IntRange:
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo > self.hi:
            raise ConfigError(f"IntRange lo {self.lo} > hi {self.hi}")

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.lo, self.hi + 1))


@dataclass(frozen=True)
class Choice:
    values: tuple
    weights: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))
        if self.weights is not None:
            w = tuple(float(x) for x in self.weights)
            if len(w) != len(self.values) or sum(w) <= 0:
                raise ConfigError("Choice weights must match values and sum > 0")
            object.__setattr__(self, "weights", w)

    def draw(self, rng: np.random.Generator):
        p = None if self.weights is None else np.asarray(self.weights) / sum(self.weights)
        return self.values[int(rng.choice(len(self.values), p=p))]


# --------------------------------------------------------------------------- config blocks

@dataclass(frozen=True)
class VideoConfig:
    frame_size: int = 300
    fps: float = 30.0
    duration_s: float = 10.0
    trace_sample_rate_hz: float = 1000.0


@dataclass(frozen=True)
class StreamsConfig:
    depth_ir_probability: float = 1.0     # P(sample includes IR + depth streams)


@dataclass(frozen=True)
class TraceConfig:
    heart_rate_bpm: Range = Range(55, 95)
    hr_variability: Range = Range(0.01, 0.05)
    resp_rate_bpm: Range = Range(10, 18)
    diastolic_mmhg: Range = Range(62, 88)
    pulse_pressure_mmhg: Range = Range(35, 60)
    cvp_mean_mmhg: Range = Range(6, 15)
    abp_upstroke_delay_s: Range = Range(0.08, 0.14)
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


@dataclass(frozen=True)
class CameraConfig:
    distance_mm: Range = Range(500, 1000)
    native_width_px: int = 3840
    hfov_deg: float = 90.0
    crop_ratio: float = 650 / 300          # rendered crop size / output size


@dataclass(frozen=True)
class GeometryConfig:
    length_mm: Range = Range(40, 80)
    separation_mm: Range = Range(10, 16)
    artery_width_mm: Range = Range(6, 9)
    vein_width_mm: Range = Range(9, 15)
    angle_deg: Range = Range(55, 125)
    centre_jitter_frac: float = 0.1
    vein_visible_fraction: Range = Range(1.0, 1.0)
    heart_to_neck_artery_m: float = 0.20
    heart_to_neck_vein_m: float = 0.15


@dataclass(frozen=True)
class AppearanceConfig:
    skin_base: Range = Range(90, 215)          # red-channel level
    skin_g_ratio: Range = Range(0.70, 0.80)
    skin_b_ratio: Range = Range(0.55, 0.68)
    monk_tone: Choice | None = None            # when set, skin colour comes from skin_rgb_by_monk
    skin_rgb_by_monk: tuple[tuple[float, float, float], ...] | None = None   # index 0 == Monk 1
    texture_sd: Range = Range(3.0, 3.0)
    vignette: Range = Range(25.0, 25.0)
    static_vessel_contrast: Range = Range(2.0, 2.0)
    ir_base: Range = Range(150, 200)
    ir_vein_contrast: Range = Range(8.0, 8.0)
    neck_radius_mm: Range = Range(50, 70)
    shading_strength: Range = Range(0.0, 0.0)  # 0 flat, 1 full Lambertian cylinder


@dataclass(frozen=True)
class PulseConfig:
    amplitude_levels: Range = Range(7, 12)     # artery, green channel, peak-to-peak
    vein_ratio: Range = Range(0.4, 0.6)        # vein amplitude / artery amplitude
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: Range = Range(0.35, 0.35)
    artery_lift_mm: Range = Range(0.3, 0.3)
    vein_lift_mm: Range = Range(0.5, 0.5)


@dataclass(frozen=True)
class IlluminationConfig:
    ambient_gain: Range = Range(1.0, 1.0)
    drift_sd: Range = Range(0.0, 0.0)          # fractional gain sd of the slow OU drift
    drift_tau_s: Range = Range(20.0, 20.0)
    flicker_amp: Range = Range(0.0, 0.0)       # fractional
    flicker_hz: Range = Range(100.0, 100.0)
    specular_amp: Range = Range(0.0, 0.0)      # levels
    specular_sigma_frac: Range = Range(0.15, 0.15)   # of frame size


@dataclass(frozen=True)
class SensorConfig:
    read_noise_sd: Range = Range(1.0, 2.0)     # levels, RGB
    shot_noise_gain: Range = Range(0.0, 0.0)   # variance per level of signal
    ir_read_noise_sd: Range = Range(2.0, 2.0)
    blur_sigma_px: Range = Range(0.0, 0.0)     # at rendered (crop) resolution
    depth_noise_mm_at_1m: Range = Range(1.6, 1.6)


@dataclass(frozen=True)
class MotionConfig:
    """Reserved for subject motion (deferred)."""


@dataclass(frozen=True)
class RhythmConfig:
    """Reserved for arrhythmia (deferred)."""


@dataclass(frozen=True)
class GeneratorConfig:
    video: VideoConfig = VideoConfig()
    streams: StreamsConfig = StreamsConfig()
    trace: TraceConfig = TraceConfig()
    camera: CameraConfig = CameraConfig()
    geometry: GeometryConfig = GeometryConfig()
    appearance: AppearanceConfig = AppearanceConfig()
    pulse: PulseConfig = PulseConfig()
    illumination: IlluminationConfig = IlluminationConfig()
    sensor: SensorConfig = SensorConfig()
    motion: MotionConfig = MotionConfig()
    rhythm: RhythmConfig = RhythmConfig()


# --------------------------------------------------------------------------- concrete params

@dataclass(frozen=True)
class VideoParams:
    frame_size: int
    fps: float
    duration_s: float

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps))


@dataclass(frozen=True)
class StreamsParams:
    has_depth_ir: bool


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
    abp_upstroke_delay_s: float = 0.10
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
    seed: int = 0

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sample_rate_hz)) + 1


@dataclass(frozen=True)
class CameraParams:
    distance_mm: float = 500.0
    native_width_px: int = 3840
    hfov_deg: float = 90.0
    crop_px: int = 650
    output_px: int = 300

    @property
    def native_scale_mm(self) -> float:
        """mm per native pixel at the centre of the frame."""
        return 2.0 * self.distance_mm * np.tan(np.deg2rad(self.hfov_deg / 2)) / self.native_width_px

    @property
    def render_scale(self) -> float:
        return self.crop_px / self.output_px

    @property
    def pixel_scale_mm(self) -> float:
        """mm per output pixel."""
        return self.native_scale_mm * self.render_scale


@dataclass(frozen=True)
class GeometryParams:
    length_mm: float = 60.0
    separation_mm: float = 13.0
    artery_width_mm: float = 7.5
    vein_width_mm: float = 12.0
    angle_deg: float = 80.0
    centre_frac_xy: tuple[float, float] = (0.5, 0.5)
    artery_side: int = 1
    vein_visible_fraction: float = 1.0
    heart_to_neck_artery_m: float = 0.20
    heart_to_neck_vein_m: float = 0.15


@dataclass(frozen=True)
class AppearanceParams:
    skin_rgb: tuple[float, float, float] = (196.0, 150.0, 124.0)
    monk_tone: int | None = None
    texture_sd: float = 3.0
    vignette: float = 25.0
    static_vessel_contrast: float = 2.0
    ir_base: float = 170.0
    ir_vein_contrast: float = 8.0
    neck_radius_mm: float = 60.0
    shading_strength: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class PulseParams:
    amplitude_levels: float = 10.0
    vein_ratio: float = 0.5
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: float = 0.35
    artery_lift_mm: float = 0.3
    vein_lift_mm: float = 0.5

    @property
    def vein_amplitude_levels(self) -> float:
        return self.amplitude_levels * self.vein_ratio


@dataclass(frozen=True)
class IlluminationParams:
    ambient_gain: float = 1.0
    drift_sd: float = 0.0
    drift_tau_s: float = 20.0
    flicker_amp: float = 0.0
    flicker_hz: float = 100.0
    specular_amp: float = 0.0
    specular_sigma_frac: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class SensorParams:
    read_noise_sd: float = 1.5
    shot_noise_gain: float = 0.0
    ir_read_noise_sd: float = 2.0
    blur_sigma_px: float = 0.0
    depth_noise_mm_at_1m: float = 1.6
    seed: int = 0


@dataclass(frozen=True)
class SampleParams:
    video: VideoParams
    streams: StreamsParams
    trace: TraceParams
    camera: CameraParams
    geometry: GeometryParams
    appearance: AppearanceParams
    pulse: PulseParams
    illumination: IlluminationParams
    sensor: SensorParams


# --------------------------------------------------------------------------- sampling

def _draw_fields(block, rng: np.random.Generator) -> dict[str, Any]:
    """Draw every Range/IntRange/Choice field of a config block; copy the rest."""
    out = {}
    for f in fields(block):
        v = getattr(block, f.name)
        out[f.name] = v.draw(rng) if isinstance(v, (Range, IntRange, Choice)) else v
    return out


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def vein_visible_fraction(posture_deg: float, cvp_mean_mmhg: float, length_mm: float, bounds: Range) -> float:
    """Fraction of the vein (from the caudal end) that pulsates visibly.

    Heuristic hydrostatics: the venous column stands ~1.36 cm per mmHg above
    the right atrium, which sits ~5 cm below the sternal angle. Supine (0 deg)
    the neck is horizontal so the whole run is below the column top. Upright,
    only the part of the neck below the column top pulsates. Always clamped to
    `bounds` so a venous signal is present in every sample.
    """
    if posture_deg <= 0:
        return bounds.hi
    column_cm = max(0.0, 1.36 * cvp_mean_mmhg - 5.0)
    neck_rise_cm = (length_mm / 10.0) * np.sin(np.deg2rad(posture_deg))
    raw = column_cm / neck_rise_cm if neck_rise_cm > 0 else 1.0
    return float(np.clip(raw, bounds.lo, bounds.hi))


def sample(config: GeneratorConfig, rng: np.random.Generator) -> SampleParams:
    """Draw one concrete parameter set. Order is fixed so seeds are reproducible."""
    v = config.video
    video = VideoParams(frame_size=v.frame_size, fps=v.fps, duration_s=v.duration_s)
    streams = StreamsParams(has_depth_ir=bool(rng.random() < config.streams.depth_ir_probability))

    t = _draw_fields(config.trace, rng)
    dia, pp = t.pop("diastolic_mmhg"), t.pop("pulse_pressure_mmhg")
    trace = TraceParams(duration_s=v.duration_s, sample_rate_hz=v.trace_sample_rate_hz,
                        diastolic_mmhg=dia, systolic_mmhg=dia + pp, seed=_seed(rng), **t)

    c = config.camera
    crop_px = int(round(v.frame_size * c.crop_ratio))
    camera = CameraParams(distance_mm=c.distance_mm.draw(rng), native_width_px=c.native_width_px,
                          hfov_deg=c.hfov_deg, crop_px=crop_px, output_px=v.frame_size)

    g = _draw_fields(config.geometry, rng)
    jitter = g.pop("centre_jitter_frac")
    g.pop("vein_visible_fraction")
    centre = (0.5 + float(rng.uniform(-jitter, jitter)), 0.5 + float(rng.uniform(-jitter, jitter)))
    side = int(rng.choice([-1, 1]))
    vis = vein_visible_fraction(trace.posture_deg, trace.cvp_mean_mmhg, g["length_mm"],
                                config.geometry.vein_visible_fraction)
    geometry = GeometryParams(centre_frac_xy=centre, artery_side=side, vein_visible_fraction=vis, **g)

    a = _draw_fields(config.appearance, rng)
    base, gr, br = a.pop("skin_base"), a.pop("skin_g_ratio"), a.pop("skin_b_ratio")
    monk, table = a.pop("monk_tone"), a.pop("skin_rgb_by_monk")
    if monk is not None and table is not None:
        skin = tuple(float(x) for x in table[int(monk) - 1])
    else:
        skin = (base, base * gr, base * br)
    appearance = AppearanceParams(skin_rgb=skin, monk_tone=None if monk is None else int(monk),
                                  seed=_seed(rng), **a)

    pulse = PulseParams(**_draw_fields(config.pulse, rng))
    illumination = IlluminationParams(seed=_seed(rng), **_draw_fields(config.illumination, rng))
    sensor = SensorParams(seed=_seed(rng), **_draw_fields(config.sensor, rng))
    return SampleParams(video, streams, trace, camera, geometry, appearance, pulse, illumination, sensor)


# --------------------------------------------------------------------------- validation

def validate(config: GeneratorConfig) -> None:
    g, v, c = config.geometry, config.video, config.camera
    if g.vein_visible_fraction.lo < 0.1:
        raise ConfigError("geometry.vein_visible_fraction lower bound must be >= 0.1 so the vein always pulses")
    if not 0.0 <= config.streams.depth_ir_probability <= 1.0:
        raise ConfigError("streams.depth_ir_probability must be in [0, 1]")
    if config.appearance.monk_tone is not None and config.appearance.skin_rgb_by_monk is None:
        raise ConfigError("appearance.monk_tone needs appearance.skin_rgb_by_monk")
    # Worst case: longest/widest vessels at the nearest camera, off-centre by the jitter.
    near = CameraParams(distance_mm=c.distance_mm.lo, native_width_px=c.native_width_px, hfov_deg=c.hfov_deg,
                        crop_px=int(round(v.frame_size * c.crop_ratio)), output_px=v.frame_size)
    extent_px = (g.length_mm.hi + g.vein_width_mm.hi + g.separation_mm.hi) / near.pixel_scale_mm
    if extent_px / 2 + g.centre_jitter_frac * v.frame_size > v.frame_size / 2:
        raise ConfigError(f"geometry cannot fit in a {v.frame_size}px frame at {c.distance_mm.lo} mm: "
                          f"extent {extent_px:.0f}px; shorten geometry.length_mm or move camera.distance_mm")


# --------------------------------------------------------------------------- overrides

def _parse_value(hint, text: str):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if text.lower() == "none":
            return None
        return _parse_value(non_none[0], text)
    if hint is Range:
        parts = [float(p) for p in text.split(",")]
        return Range(parts[0], parts[-1])
    if hint is IntRange:
        parts = [int(p) for p in text.split(",")]
        return IntRange(parts[0], parts[-1])
    if hint is Choice:
        return Choice(tuple(float(p) for p in text.split("|")))
    if hint is bool:
        return text.lower() in ("1", "true", "yes")
    if hint is int:
        return int(text)
    if hint is float:
        return float(text)
    if hint is str:
        return text
    if origin is tuple:
        return tuple(float(p) for p in text.split(","))
    raise ConfigError(f"cannot parse override for type {hint}")


def apply_override(config: GeneratorConfig, key: str, text: str) -> GeneratorConfig:
    """Return a copy of `config` with `block.field` set from `text`.

    Range: "lo,hi" or "v" (fixed). Choice: "a|b|c". Tuples: "a,b,c".
    """
    try:
        block_name, field_name = key.split(".", 1)
    except ValueError:
        raise ConfigError(f"override key must be block.field, got {key!r}")
    if block_name not in {f.name for f in fields(config)}:
        raise ConfigError(f"unknown config block {block_name!r}")
    block = getattr(config, block_name)
    hints = typing.get_type_hints(type(block))
    if field_name not in hints:
        raise ConfigError(f"unknown field {key!r}")
    try:
        value = _parse_value(hints[field_name], text)
    except (ValueError, IndexError) as e:
        raise ConfigError(f"cannot parse {text!r} for {key}: {e}") from e
    return replace(config, **{block_name: replace(block, **{field_name: value})})


# --------------------------------------------------------------------------- serialisation

def _to_jsonable(obj):
    if isinstance(obj, (Range, IntRange)):
        return {"lo": obj.lo, "hi": obj.hi}
    if isinstance(obj, Choice):
        return {"values": list(obj.values), "weights": None if obj.weights is None else list(obj.weights)}
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def config_to_dict(config: GeneratorConfig) -> dict:
    return _to_jsonable(config)


def params_to_dict(params: SampleParams) -> dict:
    return _to_jsonable(params)


def _from_jsonable(hint, value):
    if value is None:
        return None
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        return _from_jsonable([a for a in args if a is not type(None)][0], value)
    if hint is Range:
        return Range(value["lo"], value["hi"])
    if hint is IntRange:
        return IntRange(value["lo"], value["hi"])
    if hint is Choice:
        return Choice(tuple(value["values"]), None if value["weights"] is None else tuple(value["weights"]))
    if is_dataclass(hint):
        hints = typing.get_type_hints(hint)
        return hint(**{k: _from_jsonable(hints[k], v) for k, v in value.items()})
    if origin is tuple:
        inner = args[0] if args else float
        return tuple(_from_jsonable(inner, v) for v in value)
    return value


def config_from_dict(d: dict) -> GeneratorConfig:
    return _from_jsonable(GeneratorConfig, d)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 7 passed. If `test_validation_rejects_bad_configs` fails on the `fit` case, check the arithmetic: at 500 mm the pixel scale is ≈0.564 mm/px, so 400+15+16 mm ≈ 764 px, which cannot fit in 300 px.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/config.py tests/test_config.py
git commit -m "Add GeneratorConfig ranges, sampling, overrides and validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Pressure traces (`traces.py`)

**Files:**
- Create: `src/synthetic_neck/traces.py`, `tests/test_traces.py`

**Interfaces:**
- Consumes: `TraceParams` from `synthetic_neck.config`.
- Produces: `generate_trace(p: TraceParams) -> np.ndarray (N,3)` columns Time, ABP, CVP; `write_trace_csv(trace, path)`; `read_trace_csv(path) -> np.ndarray`; `TRACE_COLUMNS`; `CVP_MIN_MMHG`, `CVP_MAX_MMHG`.

- [ ] **Step 1: Write the failing tests**

`tests/test_traces.py`:
```python
import numpy as np

from synthetic_neck.config import GeneratorConfig, TraceParams, sample
from synthetic_neck.traces import generate_trace, read_trace_csv, write_trace_csv


def test_trace_shape_and_ranges():
    p = TraceParams(systolic_mmhg=130, diastolic_mmhg=80, cvp_mean_mmhg=6)
    tr = generate_trace(p)
    assert tr.shape == (10001, 3)
    t, abp, cvp = tr.T
    assert t[0] == 0 and abs(t[-1] - 10.0) < 1e-9
    assert 60 < abp.min() and abp.max() < 145
    assert 2 <= cvp.min() and cvp.max() <= 20
    assert abs(abp.max() - 130) < 4 and abs(abp.min() - 80) < 4


def test_trace_heart_rate_is_respected():
    p = TraceParams(heart_rate_bpm=90, hr_variability=0.0)
    abp = generate_trace(p)[:, 1]
    above = abp > (abp.min() + 0.6 * (abp.max() - abp.min()))
    rises = np.flatnonzero(np.diff(above.astype(int)) == 1)
    rises = rises[np.insert(np.diff(rises) > 300, 0, True)]  # 300 ms refractory vs noise
    assert abs(len(rises) - 15) <= 1


def test_cvp_a_wave_precedes_arterial_upstroke():
    p = TraceParams(heart_rate_bpm=60, hr_variability=0.0, abp_upstroke_delay_s=0.10)
    t, abp, cvp = generate_trace(p).T
    win = (t > 2.5) & (t < 3.5)      # R-wave lands at t=3.0 by construction
    t_abp_peak = t[win][np.argmax(abp[win])]
    pre = (t > t_abp_peak - 0.35) & (t < t_abp_peak - 0.05)
    t_a = t[pre][np.argmax(cvp[pre])]
    assert t_a < t_abp_peak


def test_wave_amplitudes_scale_cvp_pulse_pressure():
    small = TraceParams(hr_variability=0.0, a_wave_mmhg=1.0, v_wave_mmhg=1.0, cvp_noise_mmhg=0.0,
                        resp_cvp_swing_mmhg=0.0)
    big = TraceParams(hr_variability=0.0, a_wave_mmhg=4.0, v_wave_mmhg=4.0, cvp_noise_mmhg=0.0,
                      resp_cvp_swing_mmhg=0.0)
    pp = lambda p: np.ptp(generate_trace(p)[:, 2])
    assert pp(big) > 1.5 * pp(small)


def test_sampled_cvp_stays_within_2_to_20_mmhg():
    rng = np.random.default_rng(3)
    for _ in range(20):
        cvp = generate_trace(sample(GeneratorConfig(), rng).trace)[:, 2]
        assert 2 <= cvp.min() and cvp.max() <= 20
        assert 4 < cvp.mean() < 17


def test_trace_csv_round_trip(tmp_path):
    tr = generate_trace(TraceParams())
    write_trace_csv(tr, tmp_path / "trace.csv")
    back = read_trace_csv(tmp_path / "trace.csv")
    assert (tmp_path / "trace.csv").read_text().splitlines()[0] == "Time,ABP,CVP"
    np.testing.assert_allclose(back, tr, atol=1e-3)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_traces.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement traces.py**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_traces.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/traces.py tests/test_traces.py
git commit -m "Add ABP/CVP trace synthesis driven by TraceParams

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Vessel geometry and camera (`geometry.py`)

**Files:**
- Create: `src/synthetic_neck/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Consumes: `GeometryParams`, `CameraParams` from `config`.
- Produces:
  - `VesselGeometry` dataclass (pixel units) with `direction`, `normal`, `axis_start(which)`.
  - `vessel_geometry(gp: GeometryParams, camera: CameraParams) -> VesselGeometry` (output-pixel frame).
  - `tube_fields(g, which) -> (weight, s)`; `visibility_taper(s, length_px, fraction) -> np.ndarray`; `id_map(g) -> uint8`.
  - `arterial_pwv_m_s(map)`, `venous_pwv_m_s(cvp)`, `delay_map(s, pwv, heart_to_neck_m, pixel_scale_mm)`, `scaled(g, k)`.
  - `ID_BACKGROUND, ID_ARTERY, ID_VEIN = 0, 1, 2`.

- [ ] **Step 1: Write the failing tests**

`tests/test_geometry.py`:
```python
import numpy as np
import pytest

from synthetic_neck.config import CameraParams, GeneratorConfig, sample
from synthetic_neck.geometry import (
    VesselGeometry, delay_map, id_map, scaled, tube_fields, vessel_geometry, visibility_taper,
)


def test_camera_pixel_scale():
    cam = CameraParams(distance_mm=500)
    assert cam.native_scale_mm == pytest.approx(500 / 1920, rel=1e-6)
    assert cam.pixel_scale_mm == pytest.approx(500 / 1920 * 650 / 300, rel=1e-6)


def test_sampled_geometry_is_physical_and_fits_frame():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample(GeneratorConfig(), rng)
        scale = p.camera.pixel_scale_mm
        g = vessel_geometry(p.geometry, p.camera)
        assert g.frame_size == 300
        assert 40 <= g.length_px * scale <= 80
        assert 10 <= g.separation_px * scale <= 16
        assert 6 <= g.artery_width_px * scale <= 9 and 9 <= g.vein_width_px * scale <= 15
        assert 35 <= g.length_px <= 145
        ids = id_map(g)
        assert (ids == 1).any() and (ids == 2).any()
        assert ids[0].max() == 0 and ids[-1].max() == 0 and ids[:, 0].max() == 0 and ids[:, -1].max() == 0


def test_tube_profile_is_gaussian_with_fwhm():
    g = VesselGeometry(angle_deg=90, artery_width_px=20, centre_xy=(150, 150), separation_px=20, artery_side=1)
    w, s = tube_fields(g, "artery")
    axis_x = 160
    row = w[150]
    assert np.isclose(row[axis_x], 1.0)
    assert np.isclose(row[axis_x + 10], 0.5, atol=0.02)
    assert s[150, axis_x] == pytest.approx(35.0, abs=0.6)
    assert s[:, axis_x].min() == 0 and s[:, axis_x].max() == g.length_px


def test_visibility_taper_keeps_caudal_end_and_fades_cranial():
    s = np.linspace(0, 100, 101)
    t = visibility_taper(s, 100.0, 0.5)
    assert t[0] == pytest.approx(1.0) and t[40] == pytest.approx(1.0)
    assert t[60] < 0.6 and t[100] < 0.01
    np.testing.assert_allclose(visibility_taper(s, 100.0, 1.0), 1.0)


def test_scaled_geometry_keeps_layout():
    g = VesselGeometry(frame_size=300, length_px=70, centre_xy=(150, 140))
    k = scaled(g, 650 / 300)
    assert k.frame_size == 650 and k.length_px == pytest.approx(70 * 650 / 300)
    assert k.centre_xy[1] == pytest.approx(140 * 650 / 300)


def test_delay_grows_along_tube():
    s = np.array([0.0, 50.0, 100.0])
    d = delay_map(s, pwv_m_s=2.0, heart_to_neck_m=0.15, pixel_scale_mm=0.5)
    assert d[0] == pytest.approx(0.075) and d[2] == pytest.approx((0.15 + 0.05) / 2.0)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement geometry.py**

```python
"""Vessel layout (pixels), per-pixel weights, visibility taper and propagation delays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CameraParams, GeometryParams

ID_BACKGROUND, ID_ARTERY, ID_VEIN = 0, 1, 2


@dataclass
class VesselGeometry:
    """Two parallel straight tubes in a square (frame_size) frame.

    The caudal end (s=0, nearest the heart) is `axis_start(which)`; waves travel
    towards `start + length * direction` (cranial). Widths are FWHM in pixels.
    """
    frame_size: int = 300
    length_px: float = 70.0
    separation_px: float = 25.0
    artery_width_px: float = 16.0
    vein_width_px: float = 18.0
    angle_deg: float = 80.0          # propagation direction, 0 = +x (right), 90 = up
    centre_xy: tuple[float, float] = (150.0, 150.0)
    artery_side: int = +1
    vein_visible_fraction: float = 1.0

    @property
    def direction(self) -> np.ndarray:
        a = np.deg2rad(self.angle_deg)
        return np.array([np.cos(a), -np.sin(a)])  # image y axis points down

    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])

    def axis_start(self, which: str) -> np.ndarray:
        side = self.artery_side if which == "artery" else -self.artery_side
        c = np.asarray(self.centre_xy) + side * 0.5 * self.separation_px * self.normal
        return c - 0.5 * self.length_px * self.direction


def vessel_geometry(gp: GeometryParams, camera: CameraParams) -> VesselGeometry:
    """Convert mm anatomy to output pixels at the camera's pixel scale."""
    n, k = camera.output_px, camera.pixel_scale_mm
    return VesselGeometry(
        frame_size=n,
        length_px=gp.length_mm / k,
        separation_px=gp.separation_mm / k,
        artery_width_px=gp.artery_width_mm / k,
        vein_width_px=gp.vein_width_mm / k,
        angle_deg=gp.angle_deg,
        centre_xy=(gp.centre_frac_xy[0] * n, gp.centre_frac_xy[1] * n),
        artery_side=gp.artery_side,
        vein_visible_fraction=gp.vein_visible_fraction,
    )


def _fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / 2.3548


def tube_fields(g: VesselGeometry, which: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight, s) maps of shape (H, W): Gaussian cross-section with soft
    end taper, and distance along the axis from the caudal end clipped to [0, L]."""
    n = g.frame_size
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    start = g.axis_start(which)
    rel_x, rel_y = xs - start[0], ys - start[1]
    d = g.direction
    s_raw = rel_x * d[0] + rel_y * d[1]
    perp = rel_x * g.normal[0] + rel_y * g.normal[1]
    width = g.artery_width_px if which == "artery" else g.vein_width_px
    sigma = _fwhm_to_sigma(width)
    cross = np.exp(-0.5 * (perp / sigma) ** 2)
    overshoot = np.maximum(0.0, np.maximum(-s_raw, s_raw - g.length_px))
    taper = np.exp(-0.5 * (overshoot / sigma) ** 2)
    s = np.clip(s_raw, 0.0, g.length_px)
    return cross * taper, s


def visibility_taper(s: np.ndarray, length_px: float, fraction: float) -> np.ndarray:
    """1 over the caudal `fraction` of the tube, smooth Gaussian fall-off beyond it."""
    if fraction >= 1.0:
        return np.ones_like(s)
    edge = fraction * length_px
    width = max(0.05 * length_px, 1.0)
    return np.where(s <= edge, 1.0, np.exp(-0.5 * ((s - edge) / width) ** 2))


def id_map(g: VesselGeometry) -> np.ndarray:
    """Label map: 0 background, 1 artery, 2 vein (inside the FWHM). The vein
    label covers the whole tube even where the pulse is tapered off."""
    ids = np.zeros((g.frame_size, g.frame_size), dtype=np.uint8)
    w_v, _ = tube_fields(g, "vein")
    w_a, _ = tube_fields(g, "artery")
    ids[w_v >= 0.5] = ID_VEIN
    ids[w_a >= 0.5] = ID_ARTERY
    return ids


def arterial_pwv_m_s(mean_pressure_mmhg: float) -> float:
    """Carotid PWV vs MAP: ~7.5 m/s at 90 mmHg, +0.04 m/s per mmHg."""
    return 4.0 + 0.04 * mean_pressure_mmhg


def venous_pwv_m_s(mean_pressure_mmhg: float) -> float:
    """Jugular PWV vs mean CVP: slow (~1-3 m/s), faster as the vein distends."""
    return 0.8 + 0.15 * mean_pressure_mmhg


def delay_map(s: np.ndarray, pwv_m_s: float, heart_to_neck_m: float, pixel_scale_mm: float) -> np.ndarray:
    """Seconds between the trace (measured near the heart) and each pixel."""
    return (heart_to_neck_m + s * pixel_scale_mm * 1e-3) / pwv_m_s


def scaled(g: VesselGeometry, k: float) -> VesselGeometry:
    """The same layout expressed in a frame k times larger."""
    return VesselGeometry(
        frame_size=int(round(g.frame_size * k)),
        length_px=g.length_px * k,
        separation_px=g.separation_px * k,
        artery_width_px=g.artery_width_px * k,
        vein_width_px=g.vein_width_px * k,
        angle_deg=g.angle_deg,
        centre_xy=(g.centre_xy[0] * k, g.centre_xy[1] * k),
        artery_side=g.artery_side,
        vein_visible_fraction=g.vein_visible_fraction,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/geometry.py tests/test_geometry.py
git commit -m "Add vessel geometry, visibility taper and propagation delays

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Sensor helpers and stage (`render/camera.py`)

**Files:**
- Create: `src/synthetic_neck/render/__init__.py` (empty docstring), `src/synthetic_neck/render/camera.py`, `tests/test_render_camera.py`

**Interfaces:**
- Consumes: `SensorParams`, `CameraParams` from `config`.
- Produces:
  - `area_resample_matrix(n_in, n_out) -> (n_out, n_in)`; `gaussian_blur(img, sigma_px) -> img`; `depth_to_uint16(depth_mm) -> uint16`; `to_gray(rgb) -> uint8`; `DEPTH_UNITS_MM = 0.02`.
  - `SensorStage(params: SensorParams, camera: CameraParams, out_size: int)` with methods `rgb(img_crop_res, noise=True) -> uint8 (H,W,3)`, `ir(img, noise=True) -> uint8 (H,W)`, `depth(depth_mm, noise=True) -> float64 (H,W)`, and `.downsample(img)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_render_camera.py`:
```python
import numpy as np
import pytest

from synthetic_neck.config import CameraParams, SensorParams
from synthetic_neck.render.camera import (
    SensorStage, area_resample_matrix, depth_to_uint16, gaussian_blur, to_gray,
)


def test_area_resample_preserves_mean_and_edges():
    m = area_resample_matrix(650, 300)
    np.testing.assert_allclose(m.sum(1), 1.0)
    img = np.random.default_rng(0).random((650, 650))
    out = m @ img @ m.T
    assert out.shape == (300, 300) and abs(out.mean() - img.mean()) < 1e-9


def test_gaussian_blur_identity_and_smoothing():
    img = np.zeros((41, 41))
    img[20, 20] = 1.0
    assert gaussian_blur(img, 0.0) is img
    b = gaussian_blur(img, 2.0)
    assert b.sum() == pytest.approx(1.0, abs=1e-6)
    assert b[20, 20] < 0.1 and b[20, 22] > 0.01
    rgb = np.stack([img] * 3, -1)
    assert gaussian_blur(rgb, 1.0).shape == rgb.shape


def test_sensor_identity_when_noise_and_blur_are_zero():
    cam = CameraParams(crop_px=64, output_px=32)
    st = SensorStage(SensorParams(read_noise_sd=0.0, shot_noise_gain=0.0, ir_read_noise_sd=0.0,
                                  blur_sigma_px=0.0, depth_noise_mm_at_1m=0.0), cam, 32)
    img = np.full((64, 64, 3), 100.4)
    out = st.rgb(img)
    assert out.dtype == np.uint8 and (out == 100).all()
    assert (st.ir(np.full((64, 64), 7.6)) == 8).all()
    np.testing.assert_allclose(st.depth(np.full((64, 64), 500.0)), 500.0)


def test_sensor_noise_matches_configured_sd():
    cam = CameraParams(crop_px=64, output_px=64)
    st = SensorStage(SensorParams(read_noise_sd=2.0, shot_noise_gain=0.0, seed=3), cam, 64)
    out = st.rgb(np.full((64, 64, 3), 128.0)).astype(float)
    assert out.std() == pytest.approx(2.0, abs=0.3)
    st = SensorStage(SensorParams(read_noise_sd=0.0, shot_noise_gain=0.05, seed=3), cam, 64)
    out = st.rgb(np.full((64, 64, 3), 200.0)).astype(float)
    assert out.std() == pytest.approx(np.sqrt(0.05 * 200), abs=0.4)
    st = SensorStage(SensorParams(depth_noise_mm_at_1m=1.6, seed=3), CameraParams(distance_mm=500, crop_px=64, output_px=64), 64)
    assert st.depth(np.full((64, 64), 500.0)).std() == pytest.approx(0.4, abs=0.05)


def test_depth_and_gray_conversions():
    assert depth_to_uint16(np.array([500.0])).tolist() == [25000]
    with pytest.raises(ValueError):
        depth_to_uint16(np.array([2000.0]))
    assert to_gray(np.array([[[255, 255, 255]]], np.uint8)).tolist() == [[255]]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_render_camera.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render/__init__.py and render/camera.py**

`src/synthetic_neck/render/__init__.py`:
```python
"""Rendering stages: base scene -> pulse -> illumination -> sensor."""
```

`src/synthetic_neck/render/camera.py`:
```python
"""Sensor stage: blur, area downsample, signal-dependent noise, quantisation."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import CameraParams, SensorParams

DEPTH_UNITS_MM = 0.02  # 16-bit depth stores depth_mm / DEPTH_UNITS_MM (range 0-1310 mm)


def area_resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix that box-averages n_in samples into n_out bins."""
    m = np.zeros((n_out, n_in))
    edges = np.linspace(0, n_in, n_out + 1)
    for i in range(n_out):
        lo, hi = edges[i], edges[i + 1]
        for j in range(int(np.floor(lo)), int(np.ceil(hi))):
            m[i, j] = min(hi, j + 1) - max(lo, j)
    return m / (n_in / n_out)


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="edge")
    windows = sliding_window_view(padded, len(k), axis=axis)
    return windows @ k


def gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur over the first two axes (H, W[, C]); edge-padded."""
    if sigma_px <= 0:
        return img
    k = _gauss_kernel(sigma_px)
    return _convolve_axis(_convolve_axis(img, k, 0), k, 1)


def depth_to_uint16(depth_mm: np.ndarray) -> np.ndarray:
    q = np.rint(depth_mm / DEPTH_UNITS_MM)
    if q.max() > 65535 or q.min() < 0:
        raise ValueError(f"depth {depth_mm.min():.0f}-{depth_mm.max():.0f} mm exceeds 16-bit range")
    return q.astype(np.uint16)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma, uint8 (H, W)."""
    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


class SensorStage:
    """Turns a crop-resolution float image into an output-resolution sensor frame."""

    def __init__(self, params: SensorParams, camera: CameraParams, out_size: int):
        self.p = params
        self.camera = camera
        self._down = area_resample_matrix(camera.crop_px, out_size)
        self.rng = np.random.default_rng(params.seed)
        # Kinect-like depth noise grows with the square of distance.
        self.depth_noise_sd_mm = params.depth_noise_mm_at_1m * (camera.distance_mm / 1000.0) ** 2

    def downsample(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return np.stack([self.downsample(img[..., c]) for c in range(img.shape[-1])], axis=-1)
        return self._down @ img @ self._down.T

    def _noise(self, signal: np.ndarray, read_sd: float) -> np.ndarray:
        var = read_sd ** 2 + self.p.shot_noise_gain * np.clip(signal, 0, None)
        return self.rng.standard_normal(signal.shape) * np.sqrt(var)

    def _quantise8(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    def rgb(self, img: np.ndarray, noise: bool = True) -> np.ndarray:
        img = self.downsample(gaussian_blur(img, self.p.blur_sigma_px))
        if noise:
            img = img + self._noise(img, self.p.read_noise_sd)
        return self._quantise8(img)

    def ir(self, img: np.ndarray, noise: bool = True) -> np.ndarray:
        img = self.downsample(gaussian_blur(img, self.p.blur_sigma_px))
        if noise:
            img = img + self._noise(img, self.p.ir_read_noise_sd)
        return self._quantise8(img)

    def depth(self, depth_mm: np.ndarray, noise: bool = True) -> np.ndarray:
        d = self.downsample(depth_mm)
        if noise and self.depth_noise_sd_mm > 0:
            d = d + self.rng.standard_normal(d.shape) * self.depth_noise_sd_mm
        return d
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_render_camera.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render tests/test_render_camera.py
git commit -m "Add sensor stage: blur, area downsample, shot/read noise, quantisation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Base scene with cylindrical shading (`render/base.py`)

**Files:**
- Create: `src/synthetic_neck/render/base.py`, `tests/test_render_base.py`

**Interfaces:**
- Consumes: `AppearanceParams`, `CameraParams`; `VesselGeometry` (crop resolution) and `tube_fields`.
- Produces: `BaseScene(appearance, geometry_crop, camera, w_art, w_vein)` with attributes `rgb (n,n,3) float`, `ir (n,n) float`, `depth_mm (n,n) float`, `shading (n,n) float in (0,1]`, `perp_mm (n,n)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_render_base.py`:
```python
import numpy as np
import pytest

from synthetic_neck.config import AppearanceParams, CameraParams
from synthetic_neck.geometry import VesselGeometry, tube_fields
from synthetic_neck.render.base import BaseScene


def _scene(**kw):
    g = VesselGeometry(frame_size=100, angle_deg=90, centre_xy=(50, 50), length_px=40)
    cam = CameraParams(distance_mm=500, crop_px=100, output_px=100)
    w_a, _ = tube_fields(g, "artery")
    w_v, _ = tube_fields(g, "vein")
    return BaseScene(AppearanceParams(texture_sd=0.0, vignette=0.0, static_vessel_contrast=0.0, **kw), g, cam, w_a, w_v)


def test_flat_scene_is_skin_colour():
    s = _scene(shading_strength=0.0, skin_rgb=(200.0, 150.0, 120.0))
    assert s.rgb.shape == (100, 100, 3)
    np.testing.assert_allclose(s.rgb[..., 0], 200.0)
    np.testing.assert_allclose(s.shading, 1.0)


def test_shading_peaks_on_cylinder_axis_and_falls_off():
    s = _scene(shading_strength=1.0, neck_radius_mm=30.0)
    # Vessel direction is vertical, so the cylinder axis is the vertical centre line: shading is
    # constant down a column and falls off left/right.
    assert s.shading[50, 50] == pytest.approx(1.0, abs=1e-6)
    assert s.shading[10, 50] == pytest.approx(s.shading[90, 50])
    assert s.shading[50, 5] < s.shading[50, 25] < s.shading[50, 50]
    # Depth is nearest on the axis and the shading peak coincides with the depth minimum.
    assert np.argmin(s.depth_mm[50]) == np.argmax(s.shading[50])
    assert s.depth_mm[50, 50] == pytest.approx(500 - 30)


def test_texture_is_reproducible_from_seed():
    g = VesselGeometry(frame_size=100)
    cam = CameraParams(crop_px=100, output_px=100)
    w_a, _ = tube_fields(g, "artery"); w_v, _ = tube_fields(g, "vein")
    s1 = BaseScene(AppearanceParams(seed=5, texture_sd=2.0), g, cam, w_a, w_v)
    s2 = BaseScene(AppearanceParams(seed=5, texture_sd=2.0), g, cam, w_a, w_v)
    np.testing.assert_array_equal(s1.rgb, s2.rgb)
    assert s1.rgb[..., 1].std() > 1.0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_render_base.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render/base.py**

```python
"""Static scene: skin colour, texture, vignette, faint vessel darkening, and a
Lambertian cylinder whose axis runs along the vessels (shared with depth)."""
from __future__ import annotations

import numpy as np

from ..config import AppearanceParams, CameraParams
from ..geometry import VesselGeometry


class BaseScene:
    def __init__(self, appearance: AppearanceParams, geometry: VesselGeometry, camera: CameraParams,
                 w_art: np.ndarray, w_vein: np.ndarray):
        a, g, n = appearance, geometry, geometry.frame_size
        rng = np.random.default_rng(a.seed)
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
        r2 = ((xs - n / 2) ** 2 + (ys - n / 2) ** 2) / (n / 2) ** 2
        texture = rng.standard_normal((n, n)) * a.texture_sd

        # Cylinder: axis through the frame centre along the vessel direction.
        pixel_scale_mm = camera.native_scale_mm            # mm per rendered pixel
        self.perp_mm = ((xs - n / 2) * g.normal[0] + (ys - n / 2) * g.normal[1]) * pixel_scale_mm
        r = a.neck_radius_mm
        cos_theta = np.sqrt(np.clip(1.0 - (self.perp_mm / r) ** 2, 0.0, 1.0))
        self.shading = 1.0 - a.shading_strength * (1.0 - cos_theta)
        self.depth_mm = camera.distance_mm - np.sqrt(np.clip(r ** 2 - self.perp_mm ** 2, 0, None))

        static = -a.vignette * r2 + texture - a.static_vessel_contrast * (w_art + w_vein)
        skin = np.asarray(a.skin_rgb, dtype=np.float64)
        self.rgb = skin[None, None, :] * self.shading[..., None] + static[..., None]
        self.ir = (a.ir_base * self.shading + 0.5 * texture - 0.5 * a.vignette * r2
                   - a.ir_vein_contrast * w_vein - a.static_vessel_contrast * w_art)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_render_base.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render/base.py tests/test_render_base.py
git commit -m "Add base scene with cylindrical shading shared with depth

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Pulse stage (`render/pulse.py`)

**Files:**
- Create: `src/synthetic_neck/render/pulse.py`, `tests/test_render_pulse.py`

**Interfaces:**
- Consumes: `PulseParams`, `GeometryParams` (for heart-to-neck distances), `VesselGeometry` (crop res), `tube_fields`, `visibility_taper`, `delay_map`, `arterial_pwv_m_s`, `venous_pwv_m_s`; trace array (N,3).
- Produces: `PulseStage(pulse, geometry_params, geometry_crop, pixel_scale_mm, trace)` with attributes `w_art`, `w_vein` (vein already multiplied by the visibility taper), `delay_art`, `delay_vein`, `pwv_art`, `pwv_vein`, `map_art`, `mean_cvp`, `t`, `abp_n`, `cvp_n`; methods `pulse_maps(time_s) -> (p_art, p_vein)`, `rgb_mod(time_s) -> (n,n,3)`, `ir_mod(time_s) -> (n,n)`, `depth_lift_mm(time_s) -> (n,n)`; helper `normalise(x)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_render_pulse.py`:
```python
import numpy as np

from synthetic_neck.config import GeometryParams, PulseParams, TraceParams
from synthetic_neck.geometry import VesselGeometry, tube_fields
from synthetic_neck.render.pulse import PulseStage, normalise
from synthetic_neck.traces import generate_trace


def _stage(**geom):
    tr = generate_trace(TraceParams(heart_rate_bpm=60, hr_variability=0.0))
    g = VesselGeometry(frame_size=120, angle_deg=90, length_px=80, centre_xy=(60, 60), **geom)
    return PulseStage(PulseParams(amplitude_levels=10.0, vein_ratio=0.5), GeometryParams(), g, 0.5, tr), g, tr


def test_normalise_maps_to_half_range():
    x = np.linspace(0, 1, 1000)
    n = normalise(x)
    assert n.min() >= -0.5 and n.max() <= 0.5 and abs(n.mean()) < 0.02


def test_wave_propagates_from_caudal_to_cranial_end():
    st, g, _ = _stage()
    w, s = tube_fields(g, "vein")
    caudal = np.unravel_index(np.argmax((s < 1) * w), s.shape)
    cranial = np.unravel_index(np.argmax((s > g.length_px - 1) * w), s.shape)
    assert st.delay_vein[cranial] > st.delay_vein[caudal]
    assert st.delay_art[cranial] > st.delay_art[caudal]
    times = np.arange(0, 10, 1 / 1000)
    pv_caudal = np.interp(times - st.delay_vein[caudal], st.t, st.cvp_n)
    pv_cranial = np.interp(times - st.delay_vein[cranial], st.t, st.cvp_n)
    lag = np.argmax(np.correlate(pv_cranial - pv_cranial.mean(), pv_caudal - pv_caudal.mean(), "full")) - (len(times) - 1)
    assert lag > 0


def test_modulation_is_negative_with_pressure_and_scaled_by_amplitude():
    st, g, tr = _stage()
    w, _ = tube_fields(g, "artery")
    px = np.unravel_index(np.argmax(w), w.shape)
    times = np.arange(300) / 30.0
    green = np.array([st.rgb_mod(t)[px][1] for t in times])
    abp = np.array([np.interp(t - st.delay_art[px], st.t, tr[:, 1]) for t in times])
    assert np.corrcoef(green, abp)[0, 1] < -0.9
    assert 9 <= np.ptp(green) <= 11
    lift = np.array([st.depth_lift_mm(t)[px] for t in times])
    assert np.corrcoef(lift, abp)[0, 1] > 0.9 and 0.2 < np.ptp(lift) < 0.4


def test_vein_taper_silences_cranial_end():
    st, g, _ = _stage(vein_visible_fraction=0.5)
    w, s = tube_fields(g, "vein")
    caudal = np.unravel_index(np.argmax((s < 1) * w), s.shape)
    cranial = np.unravel_index(np.argmax((s > g.length_px - 1) * w), s.shape)
    assert st.w_vein[caudal] > 0.9 and st.w_vein[cranial] < 0.01
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_render_pulse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render/pulse.py**

```python
"""Pulse stage: pressure -> additive pixel modulation with propagation delay."""
from __future__ import annotations

import numpy as np

from ..config import GeometryParams, PulseParams
from ..geometry import (VesselGeometry, arterial_pwv_m_s, delay_map, tube_fields, venous_pwv_m_s,
                        visibility_taper)


def normalise(x: np.ndarray) -> np.ndarray:
    """Map a trace to [-0.5, 0.5] using its 1st/99th percentiles (robust to noise)."""
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo), 0, 1) - 0.5


class PulseStage:
    def __init__(self, pulse: PulseParams, gp: GeometryParams, geometry: VesselGeometry,
                 pixel_scale_mm: float, trace: np.ndarray):
        self.p = pulse
        self.t = trace[:, 0]
        self.abp_n = normalise(trace[:, 1])
        self.cvp_n = normalise(trace[:, 2])
        self.map_art = float(trace[:, 1].mean())
        self.mean_cvp = float(trace[:, 2].mean())
        self.pwv_art = arterial_pwv_m_s(self.map_art)
        self.pwv_vein = venous_pwv_m_s(self.mean_cvp)

        self.w_art, s_art = tube_fields(geometry, "artery")
        w_vein, s_vein = tube_fields(geometry, "vein")
        self.w_vein = w_vein * visibility_taper(s_vein, geometry.length_px, geometry.vein_visible_fraction)
        self.delay_art = delay_map(s_art, self.pwv_art, gp.heart_to_neck_artery_m, pixel_scale_mm)
        self.delay_vein = delay_map(s_vein, self.pwv_vein, gp.heart_to_neck_vein_m, pixel_scale_mm)
        self._gain = np.asarray(pulse.channel_gain, dtype=np.float64)

    def pulse_maps(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised pressure at every pixel of each vessel at `time_s`."""
        p_art = np.interp(time_s - self.delay_art, self.t, self.abp_n)
        p_vein = np.interp(time_s - self.delay_vein, self.t, self.cvp_n)
        return p_art, p_vein

    def _green_mod(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        # Higher pressure -> more blood volume -> more absorption -> darker.
        return -(self.p.amplitude_levels * self.w_art * p_art
                 + self.p.vein_amplitude_levels * self.w_vein * p_vein)

    def rgb_mod(self, time_s: float) -> np.ndarray:
        return self._green_mod(time_s)[..., None] * self._gain[None, None, :]

    def ir_mod(self, time_s: float) -> np.ndarray:
        return self.p.ir_gain * self._green_mod(time_s)

    def depth_lift_mm(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        return self.p.artery_lift_mm * self.w_art * p_art + self.p.vein_lift_mm * self.w_vein * p_vein
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_render_pulse.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render/pulse.py tests/test_render_pulse.py
git commit -m "Add pulse stage with propagation delay and vein visibility taper

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Illumination stage (`render/illumination.py`)

**Files:**
- Create: `src/synthetic_neck/render/illumination.py`, `tests/test_render_illumination.py`

**Interfaces:**
- Consumes: `IlluminationParams`; frame size, fps, n_frames; the base scene's `shading` map (for the specular position).
- Produces: `IlluminationStage(params, n, fps, n_frames, shading)` with `gain(i) -> float`, `specular (n,n)`, and `__call__(img, i) -> img` applying `img * gain(i) + specular` (specular is broadcast over channels).

- [ ] **Step 1: Write the failing tests**

`tests/test_render_illumination.py`:
```python
import numpy as np
import pytest

from synthetic_neck.config import IlluminationParams
from synthetic_neck.render.illumination import IlluminationStage


def _stage(**kw):
    shading = np.ones((50, 50))
    return IlluminationStage(IlluminationParams(**kw), 50, 30.0, 300, shading)


def test_identity_with_default_params():
    st = _stage()
    img = np.random.default_rng(0).random((50, 50, 3)) * 255
    np.testing.assert_array_equal(st(img, 0), img)
    np.testing.assert_array_equal(st(img, 299), img)


def test_ambient_gain_scales():
    st = _stage(ambient_gain=0.5)
    img = np.full((50, 50), 100.0)
    np.testing.assert_allclose(st(img, 7), 50.0)


def test_drift_is_slow_bounded_and_seeded():
    a = _stage(drift_sd=0.05, drift_tau_s=20.0, seed=1)
    b = _stage(drift_sd=0.05, drift_tau_s=20.0, seed=1)
    gains = np.array([a.gain(i) for i in range(300)])
    assert np.array_equal(gains, [b.gain(i) for i in range(300)])
    assert np.abs(gains - 1).max() < 0.3
    assert np.abs(np.diff(gains)).max() < 0.02      # no frame-to-frame jumps


def test_flicker_aliases_to_a_periodic_gain():
    st = _stage(flicker_amp=0.02, flicker_hz=100.0)
    gains = np.array([st.gain(i) for i in range(300)])
    # 100 Hz sampled at 30 fps never lands exactly on the sine peak, hence the tolerance.
    assert gains.max() == pytest.approx(1.02, abs=0.002) and gains.min() == pytest.approx(0.98, abs=0.002)


def test_specular_blob_sits_on_shading_peak():
    shading = np.ones((50, 50)) * 0.5
    shading[:, 30] = 1.0            # cylinder axis at column 30
    st = IlluminationStage(IlluminationParams(specular_amp=20.0, specular_sigma_frac=0.1), 50, 30.0, 300, shading)
    assert st.specular.max() == pytest.approx(20.0)
    assert np.unravel_index(np.argmax(st.specular), st.specular.shape) == (25, 30)
    img = np.zeros((50, 50, 3))
    assert st(img, 0)[25, 30, 1] == pytest.approx(20.0)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_render_illumination.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render/illumination.py**

```python
"""Illumination stage: ambient gain, slow OU drift, mains flicker, specular blob."""
from __future__ import annotations

import numpy as np

from ..config import IlluminationParams


class IlluminationStage:
    def __init__(self, params: IlluminationParams, n: int, fps: float, n_frames: int, shading: np.ndarray):
        self.p = params
        self.fps = fps
        rng = np.random.default_rng(params.seed)

        # Ornstein-Uhlenbeck drift, stationary sd = drift_sd, correlation time drift_tau_s.
        dt = 1.0 / fps
        a = np.exp(-dt / params.drift_tau_s) if params.drift_tau_s > 0 else 0.0
        x = np.zeros(n_frames)
        if params.drift_sd > 0:
            x[0] = rng.standard_normal() * params.drift_sd
            kick = params.drift_sd * np.sqrt(1 - a * a)
            for i in range(1, n_frames):
                x[i] = a * x[i - 1] + kick * rng.standard_normal()
        self._drift = x

        # Specular highlight: Gaussian blob centred on the shading peak nearest the frame centre.
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
        row = shading[n // 2]
        peak_x = int(np.argmax(row))
        sigma = params.specular_sigma_frac * n
        self.specular = params.specular_amp * np.exp(-0.5 * ((xs - peak_x) ** 2 + (ys - n / 2) ** 2) / sigma ** 2)

    def gain(self, i: int) -> float:
        t = i / self.fps
        flicker = 1.0 + self.p.flicker_amp * np.sin(2 * np.pi * self.p.flicker_hz * t)
        return float(self.p.ambient_gain * (1.0 + self._drift[i]) * flicker)

    def __call__(self, img: np.ndarray, i: int) -> np.ndarray:
        g = self.gain(i)
        spec = self.specular if img.ndim == 2 else self.specular[..., None]
        if g == 1.0 and self.p.specular_amp == 0.0:
            return img
        return img * g + spec
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_render_illumination.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render/illumination.py tests/test_render_illumination.py
git commit -m "Add illumination stage: ambient gain, OU drift, flicker, specular

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Renderer composing the stages (`render/renderer.py`)

**Files:**
- Create: `src/synthetic_neck/render/renderer.py`, `tests/test_renderer.py`

**Interfaces:**
- Consumes: `SampleParams`; `vessel_geometry`, `scaled`, `tube_fields`, `id_map`; `BaseScene`, `PulseStage`, `IlluminationStage`, `SensorStage`; trace (N,3).
- Produces: `Renderer(params: SampleParams, trace: np.ndarray)` with `n_frames`, `frame_time(i)`, `frame(i, noise=True) -> uint8 (H,W,3)`, `ir_frame(i, noise=True) -> uint8 (H,W)`, `depth_frame(i, noise=True) -> float64 mm (H,W)`, `ids -> uint8 (H,W)`, attributes `geometry` (output px), `render_geometry` (crop px), `pulse` (the `PulseStage`), `base`, `illumination`, `sensor`, `pixel_scale_mm`.

- [ ] **Step 1: Write the failing tests**

`tests/test_renderer.py`:
```python
import dataclasses

import numpy as np

from synthetic_neck.config import GeneratorConfig, apply_override, sample
from synthetic_neck.geometry import tube_fields
from synthetic_neck.render.renderer import Renderer
from synthetic_neck.traces import generate_trace


def _params(seed=0, **overrides):
    cfg = GeneratorConfig()
    for k, v in overrides.items():
        cfg = apply_override(cfg, k, v)
    return sample(cfg, np.random.default_rng(seed))


def test_frame_shapes_and_types():
    p = _params(**{"video.frame_size": "64"})
    r = Renderer(p, generate_trace(p.trace))
    assert r.n_frames == 300
    assert r.frame(0).shape == (64, 64, 3) and r.frame(0).dtype == np.uint8
    assert r.ir_frame(0).shape == (64, 64) and r.ir_frame(0).dtype == np.uint8
    d = r.depth_frame(0)
    assert d.shape == (64, 64) and d.dtype == np.float64 and 400 < d.mean() < 1000
    assert set(np.unique(r.ids)) <= {0, 1, 2}


def test_frame_modulation_darkens_with_pressure():
    p = _params(**{"video.frame_size": "100", "trace.heart_rate_bpm": "60", "trace.hr_variability": "0",
                   "appearance.texture_sd": "0", "sensor.read_noise_sd": "0", "pulse.amplitude_levels": "10"})
    tr = generate_trace(p.trace)
    r = Renderer(p, tr)
    w, _ = tube_fields(r.geometry, "artery")
    px = np.unravel_index(np.argmax(w), w.shape)
    green = np.array([r.frame(i, noise=False)[px][1] for i in range(r.n_frames)], dtype=float)
    delay = r.pulse.delay_art[int(px[0] * r.render_geometry.frame_size / 100), int(px[1] * r.render_geometry.frame_size / 100)]
    abp = np.array([np.interp(r.frame_time(i) - delay, tr[:, 0], tr[:, 1]) for i in range(r.n_frames)])
    assert np.corrcoef(green, abp)[0, 1] < -0.9
    assert 7 <= np.ptp(green) <= 13
    ir = np.array([r.ir_frame(i, noise=False)[px] for i in range(r.n_frames)], dtype=float)
    depth = np.array([r.depth_frame(i, noise=False)[px] for i in range(r.n_frames)])
    assert np.corrcoef(ir, abp)[0, 1] < -0.9 and np.ptp(ir) < np.ptp(green)
    assert np.corrcoef(depth, abp)[0, 1] < -0.9 and 0.15 < np.ptp(depth) < 0.4


def test_shading_and_depth_share_the_cylinder():
    p = _params(**{"video.frame_size": "64", "appearance.shading_strength": "1", "appearance.texture_sd": "0",
                   "appearance.vignette": "0", "sensor.read_noise_sd": "0"})
    r = Renderer(p, generate_trace(p.trace))
    rgb = r.frame(0, noise=False).astype(float)[..., 0]
    depth = r.depth_frame(0, noise=False)
    # Brightest and nearest pixels lie on the same axis line: high correlation of -depth with brightness.
    assert np.corrcoef(rgb.ravel(), -depth.ravel())[0, 1] > 0.8


def test_identity_stages_reproduce_plain_render():
    p = _params(**{"video.frame_size": "48"})
    r = Renderer(p, generate_trace(p.trace))
    a = r.frame(5, noise=False)
    b = Renderer(p, generate_trace(p.trace)).frame(5, noise=False)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement render/renderer.py**

```python
"""Compose the stages: base scene -> pulse -> illumination -> sensor."""
from __future__ import annotations

import numpy as np

from ..config import SampleParams
from ..geometry import id_map, scaled, tube_fields, vessel_geometry
from .base import BaseScene
from .camera import SensorStage
from .illumination import IlluminationStage
from .pulse import PulseStage


class Renderer:
    def __init__(self, params: SampleParams, trace: np.ndarray):
        self.params = params
        self.fps = params.video.fps
        self.n_frames = params.video.n_frames
        self.geometry = vessel_geometry(params.geometry, params.camera)          # output px
        self.render_geometry = scaled(self.geometry, params.camera.render_scale)  # crop px
        self.pixel_scale_mm = params.camera.native_scale_mm                       # mm per rendered px
        self.ids = id_map(self.geometry)

        g = self.render_geometry
        w_art, _ = tube_fields(g, "artery")
        w_vein, _ = tube_fields(g, "vein")
        self.base = BaseScene(params.appearance, g, params.camera, w_art, w_vein)
        self.pulse = PulseStage(params.pulse, params.geometry, g, self.pixel_scale_mm, trace)
        self.illumination = IlluminationStage(params.illumination, g.frame_size, self.fps, self.n_frames,
                                              self.base.shading)
        self.sensor = SensorStage(params.sensor, params.camera, params.video.frame_size)

    def frame_time(self, i: int) -> float:
        return i / self.fps

    def frame(self, i: int, noise: bool = True) -> np.ndarray:
        """RGB uint8 (H, W, 3) at output resolution."""
        img = self.base.rgb + self.pulse.rgb_mod(self.frame_time(i))
        img = self.illumination(img, i)
        return self.sensor.rgb(img, noise=noise)

    def ir_frame(self, i: int, noise: bool = True) -> np.ndarray:
        """Near-infrared uint8 (H, W)."""
        img = self.base.ir + self.pulse.ir_mod(self.frame_time(i))
        img = self.illumination(img, i)
        return self.sensor.ir(img, noise=noise)

    def depth_frame(self, i: int, noise: bool = True) -> np.ndarray:
        """Depth in mm, float64 (H, W). Higher pressure -> skin lifts -> nearer."""
        d = self.base.depth_mm - self.pulse.depth_lift_mm(self.frame_time(i))
        return self.sensor.depth(d, noise=noise)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: 4 passed. If the darkening test's peak-to-peak lands just outside 7–13, the culprit is the area downsample averaging the tube edge at 100 px; widen the frame to 150 rather than loosening the bound.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render/renderer.py tests/test_renderer.py
git commit -m "Add Renderer composing base, pulse, illumination and sensor stages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Presets `lesson` and `benchmark` (`presets.py`)

**Files:**
- Create: `src/synthetic_neck/presets.py`, `tests/test_presets.py`

**Interfaces:**
- Consumes: `GeneratorConfig` and blocks, `Range`, `Choice`, `validate`, `sample`.
- Produces: `lesson() -> GeneratorConfig`, `benchmark() -> GeneratorConfig`, `neckflix(priors_path=DEFAULT_PRIORS) -> GeneratorConfig` (implemented in Task 16; here it raises `FileNotFoundError` pointing at `synthetic-neck calibrate` when the file is missing), `PRESETS: dict[str, Callable[[], GeneratorConfig]]`, `get_preset(name) -> GeneratorConfig`, `DEFAULT_PRIORS: Path`, `MONK_SKIN_RGB: tuple[tuple[float,float,float], ...]` (10 entries).

- [ ] **Step 1: Write the failing tests**

`tests/test_presets.py`:
```python
import numpy as np
import pytest

from synthetic_neck.config import sample, validate
from synthetic_neck.presets import MONK_SKIN_RGB, PRESETS, benchmark, get_preset, lesson, neckflix


def test_preset_names():
    assert set(PRESETS) == {"lesson", "benchmark", "neckflix"}
    with pytest.raises(KeyError):
        get_preset("nope")


@pytest.mark.parametrize("name", ["lesson", "benchmark"])
def test_presets_validate_and_sample_50_seeds(name):
    cfg = get_preset(name)
    validate(cfg)
    for seed in range(50):
        p = sample(cfg, np.random.default_rng(seed))
        assert p.geometry.vein_visible_fraction >= cfg.geometry.vein_visible_fraction.lo
        assert p.pulse.amplitude_levels > 0


def test_amplitude_targets():
    assert lesson().pulse.amplitude_levels.lo == 7 and lesson().pulse.amplitude_levels.hi == 12
    assert benchmark().pulse.amplitude_levels.lo == 2 and benchmark().pulse.amplitude_levels.hi == 5
    assert lesson().geometry.vein_visible_fraction.lo == 1.0
    assert benchmark().geometry.vein_visible_fraction.lo == 0.6


def test_lesson_has_no_illumination_effects():
    il = lesson().illumination
    assert il.drift_sd.hi == 0 and il.flicker_amp.hi == 0 and il.specular_amp.hi == 0


def test_neckflix_without_priors_points_at_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        neckflix(tmp_path / "missing.json")


def test_monk_table_is_monotonic_in_brightness():
    lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in MONK_SKIN_RGB]
    assert len(MONK_SKIN_RGB) == 10 and all(a > b for a, b in zip(lum, lum[1:]))
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_presets.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement presets.py**

```python
"""Named presets: fully populated GeneratorConfigs for the three use cases."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import (AppearanceConfig, GeneratorConfig, GeometryConfig, IlluminationConfig, PulseConfig,
                     Range, SensorConfig)

DEFAULT_PRIORS = Path(__file__).resolve().parents[2] / "priors" / "neckflix.json"

# Approximate sRGB of the ten Monk Skin Tone swatches (index 0 == Monk 1). The
# neckflix preset replaces these with measured per-tone skin colour when the
# priors file has them.
MONK_SKIN_RGB: tuple[tuple[float, float, float], ...] = (
    (246, 237, 228), (243, 231, 219), (247, 234, 208), (234, 218, 186), (215, 189, 150),
    (160, 126, 86), (130, 92, 67), (96, 65, 52), (58, 49, 42), (41, 36, 32),
)


def lesson() -> GeneratorConfig:
    """Clean, obvious signal: large amplitude, flat lighting, whole vein pulsing."""
    return GeneratorConfig(
        pulse=PulseConfig(amplitude_levels=Range(7, 12)),
        geometry=GeometryConfig(vein_visible_fraction=Range(1.0, 1.0)),
        appearance=AppearanceConfig(shading_strength=Range(0.0, 0.0)),
        illumination=IlluminationConfig(),
        sensor=SensorConfig(read_noise_sd=Range(1.0, 2.0)),
    )


def benchmark() -> GeneratorConfig:
    """Moderate amplitude with controlled degradations turned on at modest levels."""
    return GeneratorConfig(
        pulse=PulseConfig(amplitude_levels=Range(2, 5)),
        geometry=GeometryConfig(vein_visible_fraction=Range(0.6, 1.0)),
        appearance=AppearanceConfig(shading_strength=Range(0.3, 0.8)),
        illumination=IlluminationConfig(
            ambient_gain=Range(0.85, 1.15), drift_sd=Range(0.0, 0.02), drift_tau_s=Range(10, 40),
            flicker_amp=Range(0.0, 0.01), specular_amp=Range(0.0, 15.0)),
        sensor=SensorConfig(read_noise_sd=Range(1.0, 3.0), shot_noise_gain=Range(0.0, 0.03),
                            blur_sigma_px=Range(0.0, 1.5)),
    )


def neckflix(priors_path: Path = DEFAULT_PRIORS) -> GeneratorConfig:
    """Ranges calibrated from the Neckflix dataset (see calibrate.py)."""
    priors_path = Path(priors_path)
    if not priors_path.exists():
        raise FileNotFoundError(
            f"{priors_path} not found; run `synthetic-neck calibrate --root <Neckflix dir> --out {priors_path}`")
    from .calibrate import config_from_priors   # implemented in Task 16
    return config_from_priors(priors_path)


PRESETS: dict[str, Callable[[], GeneratorConfig]] = {"lesson": lesson, "benchmark": benchmark, "neckflix": neckflix}


def get_preset(name: str) -> GeneratorConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_presets.py -v`
Expected: 7 passed (the `neckflix` import of `calibrate` is lazy, so the missing-file test passes before Task 15 exists).

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/presets.py tests/test_presets.py
git commit -m "Add lesson and benchmark presets and Monk skin table

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Sample and dataset generation (`generate.py`)

**Files:**
- Create: `src/synthetic_neck/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `GeneratorConfig`, `sample`, `validate`, `config_to_dict`, `params_to_dict`; `generate_trace`, `write_trace_csv`, `read_trace_csv`; `Renderer`; `MkvWriter`, `mux`, `require_ffmpeg`; `depth_to_uint16`, `to_gray`, `DEPTH_UNITS_MM`.
- Produces:
  - `generate_sample(config, seed: int, out_dir: Path, preset: str = "custom") -> dict` (the metadata).
  - `generate_dataset(config, out_root: Path, n: int, start: int, base_seed: int, preset: str, overrides: list[str], jobs: int = 1) -> list[tuple[int, Exception | None]]` writing `dataset.json`.
  - `SampleFailed(Exception)` with `.index` and `.seed`.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:
```python
import json

import numpy as np
from conftest import needs_ffmpeg

from synthetic_neck.config import GeneratorConfig, apply_override
from synthetic_neck.generate import generate_dataset, generate_sample
from synthetic_neck.render.camera import to_gray
from synthetic_neck.video import read_mkv


def _small():
    return apply_override(GeneratorConfig(), "video.frame_size", "64")


@needs_ffmpeg
def test_generate_sample_end_to_end(tmp_path):
    meta = generate_sample(_small(), seed=7, out_dir=tmp_path / "1", preset="lesson")
    d = tmp_path / "1"
    assert {p.name for p in d.iterdir()} == {"trace.csv", "gray_video.mkv", "rgbid_video.mkv",
                                             "vessel_ids.npy", "metadata.json"}
    assert meta["n_frames"] == 300 and meta["preset"] == "lesson" and meta["seed"] == 7
    assert meta["streams"] == ["rgb", "ir", "depth"]
    rgb = read_mkv(d / "rgbid_video.mkv", "rgb", stream=0)
    gray = read_mkv(d / "gray_video.mkv", "gray")
    assert rgb.shape == (300, 64, 64, 3) and gray.shape == (300, 64, 64)
    np.testing.assert_array_equal(gray, np.stack([to_gray(f) for f in rgb]))
    ir = read_mkv(d / "rgbid_video.mkv", "gray", stream=1)
    depth = read_mkv(d / "rgbid_video.mkv", "gray16", stream=2)
    assert ir.shape == (300, 64, 64) and depth.dtype == np.uint16
    assert 400 < (depth * 0.02).mean() < 1000
    ids = np.load(d / "vessel_ids.npy")
    assert set(np.unique(ids)) <= {0, 1, 2}
    saved = json.loads((d / "metadata.json").read_text())
    assert saved["config"]["video"]["frame_size"] == 64
    assert saved["params"]["trace"]["heart_rate_bpm"] == meta["params"]["trace"]["heart_rate_bpm"]
    assert 0.5 < saved["derived"]["pixel_scale_mm"] < 1.2


@needs_ffmpeg
def test_sample_without_depth_ir_streams(tmp_path):
    cfg = apply_override(_small(), "streams.depth_ir_probability", "0")
    meta = generate_sample(cfg, seed=1, out_dir=tmp_path / "1")
    assert meta["streams"] == ["rgb"]
    import subprocess
    n = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(tmp_path / "1" / "rgbid_video.mkv")], capture_output=True, text=True).stdout.split()
    assert len(n) == 1


@needs_ffmpeg
def test_generate_dataset_writes_index_and_reports_failures(tmp_path, monkeypatch):
    import synthetic_neck.generate as gen
    real = gen.generate_sample

    def flaky(config, seed, out_dir, preset="custom"):
        if seed == 101:
            raise RuntimeError("boom")
        return real(config, seed, out_dir, preset)

    monkeypatch.setattr(gen, "generate_sample", flaky)
    results = gen.generate_dataset(_small(), tmp_path, n=2, start=1, base_seed=100, preset="lesson",
                                   overrides=["video.frame_size=64"], jobs=1)
    assert [(i, e is None) for i, e in results] == [(1, False), (2, True)]
    assert not (tmp_path / "1").exists() and (tmp_path / "2" / "metadata.json").exists()
    idx = json.loads((tmp_path / "dataset.json").read_text())
    assert idx["preset"] == "lesson" and idx["overrides"] == ["video.frame_size=64"] and idx["base_seed"] == 100
    assert idx["failed"] == [1]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement generate.py**

```python
"""Generate one sample or a whole dataset."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import GeneratorConfig, config_to_dict, params_to_dict, sample, validate
from .render.camera import DEPTH_UNITS_MM, depth_to_uint16, to_gray
from .render.renderer import Renderer
from .traces import generate_trace, read_trace_csv, write_trace_csv
from .video import MkvWriter, mux, require_ffmpeg


class SampleFailed(Exception):
    def __init__(self, index: int, seed: int, cause: Exception):
        super().__init__(f"sample {index} (seed {seed}) failed: {cause!r}")
        self.index, self.seed, self.cause = index, seed, cause


def generate_sample(config: GeneratorConfig, seed: int, out_dir: Path, preset: str = "custom") -> dict:
    """Render one sample into `out_dir` and return its metadata dict."""
    validate(config)
    require_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = sample(config, np.random.default_rng(seed))

    # 1. Ground-truth trace first; the video is rendered from the CSV on disk.
    write_trace_csv(generate_trace(params.trace), out_dir / "trace.csv")
    trace = read_trace_csv(out_dir / "trace.csv")

    # 2. Render every stream in one pass.
    r = Renderer(params, trace)
    n = params.video.frame_size
    size, fps = (n, n), params.video.fps
    has_di = params.streams.has_depth_ir
    rgb_tmp, ir_tmp, depth_tmp = (out_dir / f"_{s}.mkv" for s in ("rgb", "ir", "depth"))
    writers = [MkvWriter(rgb_tmp, size, "rgb", fps), MkvWriter(out_dir / "gray_video.mkv", size, "gray", fps)]
    if has_di:
        writers += [MkvWriter(ir_tmp, size, "gray", fps), MkvWriter(depth_tmp, size, "gray16", fps)]
    try:
        for i in range(r.n_frames):
            rgb = r.frame(i)
            writers[0].write(rgb)
            writers[1].write(to_gray(rgb))
            if has_di:
                writers[2].write(r.ir_frame(i))
                writers[3].write(depth_to_uint16(r.depth_frame(i)))
    finally:
        for w in writers:
            w.close()
    streams = ["rgb", "ir", "depth"] if has_di else ["rgb"]
    parts = [rgb_tmp, ir_tmp, depth_tmp] if has_di else [rgb_tmp]
    mux(parts, out_dir / "rgbid_video.mkv", titles=streams)
    for tmp in parts:
        tmp.unlink()
    np.save(out_dir / "vessel_ids.npy", r.ids)

    g, cam, pulse = r.geometry, params.camera, r.pulse
    meta = {
        "seed": seed,
        "preset": preset,
        "version": __version__,
        "frame_size": n,
        "fps": fps,
        "n_frames": r.n_frames,
        "streams": streams,
        "config": config_to_dict(config),
        "params": params_to_dict(params),
        "geometry_px": {"length_px": g.length_px, "separation_px": g.separation_px,
                        "artery_width_px": g.artery_width_px, "vein_width_px": g.vein_width_px,
                        "centre_xy": list(g.centre_xy), "angle_deg": g.angle_deg},
        "derived": {
            "pixel_scale_mm": cam.pixel_scale_mm,
            "mean_abp_mmhg": pulse.map_art,
            "mean_cvp_mmhg": pulse.mean_cvp,
            "arterial_pwv_m_s": pulse.pwv_art,
            "venous_pwv_m_s": pulse.pwv_vein,
            "artery_delay_range_s": [float(pulse.delay_art.min()), float(pulse.delay_art.max())],
            "vein_delay_range_s": [float(pulse.delay_vein.min()), float(pulse.delay_vein.max())],
            "vein_visible_fraction": params.geometry.vein_visible_fraction,
        },
        "rgbid_streams": {str(i): s for i, s in enumerate(streams)},
        "depth_units_mm": DEPTH_UNITS_MM,
        "vessel_ids": {"file": "vessel_ids.npy", "0": "background", "1": "artery", "2": "vein"},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _one(args) -> tuple[int, Exception | None]:
    config, index, seed, out_dir, preset = args
    try:
        generate_sample(config, seed, out_dir, preset)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller
        shutil.rmtree(out_dir, ignore_errors=True)
        return index, e


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:               # noqa: BLE001 - not a git checkout
        return None


def generate_dataset(config: GeneratorConfig, out_root: Path, n: int, start: int, base_seed: int,
                     preset: str, overrides: list[str], jobs: int = 1) -> list[tuple[int, Exception | None]]:
    """Generate samples start..start+n-1 (seed = base_seed + i) and write dataset.json."""
    validate(config)
    require_ffmpeg()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_args = [(config, i, base_seed + i, out_root / str(i), preset) for i in range(start, start + n)]
    if jobs <= 1:
        results = [_one(a) for a in jobs_args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_one, jobs_args))
    index = {
        "preset": preset,
        "overrides": list(overrides),
        "base_seed": base_seed,
        "start": start,
        "n": n,
        "version": __version__,
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config_to_dict(config),
        "failed": [i for i, e in results if e is not None],
    }
    (out_root / "dataset.json").write_text(json.dumps(index, indent=2))
    for i, e in results:
        if e is not None:
            print(f"sample {i} failed: {e!r}", file=sys.stderr)
    return results
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 3 passed. Note `_one` calls `generate_sample` through the module global so the monkeypatch in the third test takes effect; keep it that way.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/generate.py tests/test_generate.py
git commit -m "Add sample and dataset generation with metadata and failure reporting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: FFT inspection tool (`inspect.py`)

**Files:**
- Create: `src/synthetic_neck/inspect.py`, `tests/test_inspect.py`

**Interfaces:**
- Consumes: `read_mkv`, `DEPTH_UNITS_MM`, `generate_sample`, presets.
- Produces: `load_channels(sample_dir) -> dict[str, np.ndarray]`, `fft_maps(sample_dir) -> (power, phase, hr_bin_hz)`, `vessel_phase_summary(phase, ids) -> dict[str, tuple]`, `vessel_power_ratio(power, ids) -> dict[str, tuple[float, float]]` (artery/background, vein/background), `plot_maps(sample_dir, out_path=None) -> Path` (needs matplotlib), `inspect_root(root, samples) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_inspect.py`:
```python
import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.config import apply_override
from synthetic_neck.generate import generate_sample
from synthetic_neck.inspect import fft_maps, vessel_phase_summary, vessel_power_ratio
from synthetic_neck.presets import get_preset


@needs_ffmpeg
@pytest.mark.parametrize("name", ["lesson", "benchmark", "neckflix"])
def test_fft_maps_localise_vessels_for_each_preset(tmp_path, name):
    if name == "neckflix":
        from synthetic_neck.presets import DEFAULT_PRIORS
        if not DEFAULT_PRIORS.exists():
            pytest.skip("priors/neckflix.json not generated yet (Task 16)")
    # neckflix amplitudes are near the noise floor, so give it more pixels per vessel.
    cfg = apply_override(get_preset(name), "video.frame_size", "96" if name == "neckflix" else "64")
    generate_sample(cfg, seed=11, out_dir=tmp_path / "1", preset=name)
    power, phase, f_hz = fft_maps(tmp_path / "1")
    ids = np.load(tmp_path / "1" / "vessel_ids.npy")
    ratios = vessel_power_ratio(power, ids)
    for ch in ("R", "G", "B", "IR"):
        assert ratios[ch][0] > 1.5, (ch, ratios[ch])      # artery vs background
        assert ratios[ch][1] > 1.2, (ch, ratios[ch])      # vein vs background
    assert ratios["Depth (mm)"][0] > 1.2
    diffs = [abs(v[2]) for v in vessel_phase_summary(phase, ids).values()]
    assert all(d > 0.3 for d in diffs)


@needs_ffmpeg
def test_plot_maps_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from synthetic_neck.inspect import plot_maps
    cfg = apply_override(get_preset("lesson"), "video.frame_size", "48")
    generate_sample(cfg, seed=2, out_dir=tmp_path / "1")
    assert plot_maps(tmp_path / "1").exists()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_inspect.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement inspect.py**

```python
"""QA tool: per-pixel FFT power and phase maps for every channel of a sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render.camera import DEPTH_UNITS_MM
from .video import read_mkv


def load_channels(sample_dir: Path) -> dict[str, np.ndarray]:
    meta = json.loads((sample_dir / "metadata.json").read_text())
    p = sample_dir / "rgbid_video.mkv"
    rgb = read_mkv(p, "rgb", 0).astype(np.float64)
    out = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
    if "ir" in meta["streams"]:
        out["IR"] = read_mkv(p, "gray", meta["streams"].index("ir")).astype(np.float64)
        out["Depth (mm)"] = read_mkv(p, "gray16", meta["streams"].index("depth")).astype(np.float64) * DEPTH_UNITS_MM
    return out


def fft_maps(sample_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Return ({channel: power}, {channel: phase}, hr_bin_hz). Power is the peak |FFT|^2
    within ±1 bin of the heart-rate bin; phase is the FFT angle at that bin."""
    meta = json.loads((sample_dir / "metadata.json").read_text())
    fps, hr_hz = meta["fps"], meta["params"]["trace"]["heart_rate_bpm"] / 60.0
    chans = load_channels(sample_dir)
    n = next(iter(chans.values())).shape[0]
    freqs = np.fft.rfftfreq(n, 1 / fps)
    k = int(np.argmin(np.abs(freqs - hr_hz)))
    win = np.hanning(n)[:, None, None]
    power, phase = {}, {}
    for name, v in chans.items():
        spec = np.fft.rfft((v - v.mean(0)) * win, axis=0)
        power[name] = np.abs(spec[k - 1:k + 2]).max(0) ** 2
        phase[name] = np.angle(spec[k])
    return power, phase, float(freqs[k])


def vessel_phase_summary(phase: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Circular-mean phase over artery / vein masks and their difference, per channel."""
    out = {}
    for name, ph in phase.items():
        a = np.angle(np.exp(1j * ph[ids == 1]).mean())
        v = np.angle(np.exp(1j * ph[ids == 2]).mean())
        out[name] = (float(a), float(v), float(np.angle(np.exp(1j * (v - a)))))
    return out


def vessel_power_ratio(power: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, tuple[float, float]]:
    """(artery/background, vein/background) mean power ratios per channel."""
    out = {}
    for name, pw in power.items():
        bg = pw[ids == 0].mean()
        out[name] = (float(pw[ids == 1].mean() / bg), float(pw[ids == 2].mean() / bg))
    return out


def plot_maps(sample_dir: Path, out_path: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power, phase, f_hz = fft_maps(sample_dir)
    fig, ax = plt.subplots(2, len(power), figsize=(4 * len(power), 8), squeeze=False)
    for j, name in enumerate(power):
        im = ax[0, j].imshow(power[name], cmap="viridis")
        ax[0, j].set_title(f"{name}: peak power at HR ({f_hz:.1f}±0.1 Hz)")
        fig.colorbar(im, ax=ax[0, j], fraction=0.046)
        im = ax[1, j].imshow(phase[name], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, j].set_title(f"{name}: phase at {f_hz:.1f} Hz (rad)")
        fig.colorbar(im, ax=ax[1, j], fraction=0.046)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{sample_dir.name} — per-pixel FFT (Hann window, DC removed)")
    fig.tight_layout()
    out_path = out_path or sample_dir / "fft_maps.png"
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    return out_path


def inspect_root(root: Path, samples: list[str] | None = None) -> None:
    names = samples or sorted((p.name for p in root.iterdir() if p.is_dir()), key=int)
    for name in names:
        d = root / name
        out = plot_maps(d)
        power, phase, _ = fft_maps(d)
        ids = np.load(d / "vessel_ids.npy")
        s = vessel_phase_summary(phase, ids)["G"]
        r = vessel_power_ratio(power, ids)["G"]
        print(f"{name}: {out.name}  G power artery x{r[0]:.1f} vein x{r[1]:.1f}  "
              f"phase artery {s[0]:+.2f} vein {s[1]:+.2f} diff {s[2]:+.2f} rad")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inspect.py -v`
Expected: 3 passed, 1 skipped (neckflix until Task 16). The benchmark preset at 64 px is the marginal case; if its vein/background ratio in a colour channel falls under 1.2, raise the test frame size to 96 rather than lowering the threshold.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/inspect.py tests/test_inspect.py
git commit -m "Add FFT inspection tool with per-preset vessel detectability test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Command-line interface (`cli.py`)

**Files:**
- Create: `src/synthetic_neck/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_preset`, `apply_override`, `validate`, `ConfigError`, `generate_dataset`, `inspect_root`, `run_calibration` (Task 15; imported lazily inside the subcommand).
- Produces: `build_config(preset: str, sets: list[str]) -> GeneratorConfig`, `main(argv=None) -> int`.

CLI shape:
```
synthetic-neck generate --preset lesson --n 5 --start 1 --seed 2026 --out data/synthetic_necks --jobs 1 --set pulse.amplitude_levels=0.5,2
synthetic-neck inspect --root data/synthetic_necks [--samples 1 2]
synthetic-neck calibrate --root "/Volumes/Blue 4TB/CVP/Dataset/Neckflix" --out priors/neckflix.json --n-recordings 30 --seed 0
```

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import json

import pytest
from conftest import needs_ffmpeg

from synthetic_neck.cli import build_config, main
from synthetic_neck.config import ConfigError, Range


def test_build_config_applies_preset_and_overrides():
    cfg = build_config("lesson", ["pulse.amplitude_levels=1,2", "video.frame_size=32"])
    assert cfg.pulse.amplitude_levels == Range(1, 2) and cfg.video.frame_size == 32
    with pytest.raises(ConfigError):
        build_config("lesson", ["pulse.nope=1"])
    with pytest.raises(ConfigError, match="key=value"):
        build_config("lesson", ["pulse.amplitude_levels"])


def test_bad_override_exits_nonzero_before_rendering(tmp_path, capsys):
    rc = main(["generate", "--preset", "lesson", "--n", "1", "--out", str(tmp_path), "--set", "pulse.nope=1"])
    assert rc == 2 and "unknown field" in capsys.readouterr().err
    assert not (tmp_path / "1").exists()


@needs_ffmpeg
def test_generate_command_writes_samples(tmp_path):
    rc = main(["generate", "--preset", "lesson", "--n", "2", "--seed", "5", "--out", str(tmp_path),
               "--set", "video.frame_size=32"])
    assert rc == 0
    assert (tmp_path / "1" / "metadata.json").exists() and (tmp_path / "2" / "metadata.json").exists()
    idx = json.loads((tmp_path / "dataset.json").read_text())
    assert idx["preset"] == "lesson" and idx["overrides"] == ["video.frame_size=32"]
    assert json.loads((tmp_path / "1" / "metadata.json").read_text())["seed"] == 6
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement cli.py**

```python
"""`synthetic-neck` command line: generate | inspect | calibrate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, GeneratorConfig, apply_override, validate
from .presets import PRESETS, get_preset


def build_config(preset: str, sets: list[str]) -> GeneratorConfig:
    cfg = get_preset(preset)
    for item in sets:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, text = item.split("=", 1)
        cfg = apply_override(cfg, key.strip(), text.strip())
    validate(cfg)
    return cfg


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synthetic-neck", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="render a dataset of synthetic samples")
    g.add_argument("--preset", choices=sorted(PRESETS), default="lesson")
    g.add_argument("--set", dest="sets", action="append", default=[], metavar="block.field=value",
                   help="override a config field; Range as lo,hi or v; Choice as a|b|c")
    g.add_argument("--out", type=Path, default=Path("data/synthetic_necks"))
    g.add_argument("--n", type=int, default=5)
    g.add_argument("--start", type=int, default=1)
    g.add_argument("--seed", type=int, default=2026, help="base seed; sample i uses seed+i")
    g.add_argument("--jobs", type=int, default=1)

    i = sub.add_parser("inspect", help="write fft_maps.png and print vessel power/phase per sample")
    i.add_argument("--root", type=Path, default=Path("data/synthetic_necks"))
    i.add_argument("--samples", nargs="*")

    c = sub.add_parser("calibrate", help="mine the Neckflix dataset for priors (offline)")
    c.add_argument("--root", type=Path, required=True, help="Neckflix dataset directory")
    c.add_argument("--out", type=Path, default=Path("priors/neckflix.json"))
    c.add_argument("--n-recordings", type=int, default=30)
    c.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            from .generate import generate_dataset
            cfg = build_config(args.preset, args.sets)
            results = generate_dataset(cfg, args.out, n=args.n, start=args.start, base_seed=args.seed,
                                       preset=args.preset, overrides=args.sets, jobs=args.jobs)
            failed = [i for i, e in results if e is not None]
            print(f"wrote {len(results) - len(failed)}/{len(results)} samples to {args.out}")
            return 1 if failed else 0
        if args.command == "inspect":
            from .inspect import inspect_root
            inspect_root(args.root, args.samples)
            return 0
        if args.command == "calibrate":
            from .calibrate import run_calibration
            run_calibration(args.root, args.out, n_recordings=args.n_recordings, seed=args.seed)
            print(f"wrote {args.out}")
            return 0
    except (ConfigError, FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py -v && uv run synthetic-neck generate --help`
Expected: 3 passed and the help text prints.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/cli.py tests/test_cli.py
git commit -m "Add synthetic-neck CLI with presets and --set overrides

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Neckflix calibration (`calibrate.py`)

**Files:**
- Create: `src/synthetic_neck/calibrate.py`, `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `read_mkv`-style ffmpeg decoding, `gaussian_blur`, `MkvWriter` (tests only), `TraceParams`/`generate_trace`/`beat_onsets` (tests only).
- Produces:
  - `quantiles(values) -> dict` with keys `p05,p25,p50,p75,p95`; `histogram(values) -> dict[str, float]` normalised weights.
  - `read_info(root) -> list[dict]`; `population_priors(rows) -> dict`.
  - `detect_r_peaks(ecg, fs_hz) -> np.ndarray` (sample indices).
  - `waveform_priors_for_recording(trace_csv: Path) -> dict`; `waveform_priors(per_recording: list[dict]) -> dict`.
  - `read_frames(path, fmt, start_s, n) -> np.ndarray`; `skin_mask(rgb) -> np.ndarray[bool]`.
  - `appearance_priors_for_recording(rec_dir: Path, n_frames=10) -> dict`; `appearance_priors(per_recording, monk_by_dir) -> dict`.
  - `run_calibration(root: Path, out: Path, n_recordings=30, seed=0) -> dict`.
  - `config_from_priors(path: Path) -> GeneratorConfig` (used by `presets.neckflix`).
  - `PRIORS_SCHEMA_VERSION = 1`.

Neckflix facts used here: `data/<Recording_Directory>/trace_data.csv` has columns `Time (s),CVP (mmHg),ECG (mV)` at 20 kHz (ECG R-peaks positive, raw units); `K1_RGB.mkv` is 650×650 bgr0 30 fps; `K1_IR.mkv` is 650×650 gray16le; `dataset_info.csv` columns include `Recording_Directory, K1_RGB.mkv, K1_IR.mkv, ECG (mV), CVP (mmHg), Sex, Age (years), Neck_Circumference_Mid (cm), Skin_Tone_Clinician, Skin_Tone_Recorder, BP_Sys (mmHG), BP_Dia (mmHg), JVP Height Estimate (cm), Average_BPM`. Posture is the second-last `_`-separated token of the directory name; `_D`/`_N` suffix is depth+IR availability.

- [ ] **Step 1: Write the failing tests**

`tests/test_calibrate.py`:
```python
import csv
import json

import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.calibrate import (
    config_from_priors, detect_r_peaks, histogram, population_priors, quantiles, run_calibration,
    skin_mask, waveform_priors_for_recording,
)
from synthetic_neck.config import TraceParams, validate
from synthetic_neck.traces import generate_trace
from synthetic_neck.video import MkvWriter

INFO_COLUMNS = ["Recording_Directory", "Participant_ID", "K1_RGB.mkv", "K1_IR.mkv", "ECG (mV)", "CVP (mmHg)", "Sex",
                "Age (years)", "Neck_Circumference_Mid (cm)", "Skin_Tone_Clinician", "Skin_Tone_Recorder",
                "BP_Sys (mmHG)", "BP_Dia (mmHg)", "JVP Height Estimate (cm)", "Average_BPM"]


def _row(pid, tone, posture, jvp, bpm, depth="D"):
    name = f"P{pid:03d}_S01_R1_{posture}_{depth}"
    return dict(zip(INFO_COLUMNS, [name, str(pid), "TRUE", "TRUE" if depth == "D" else "FALSE", "TRUE", "TRUE", "M",
                                   "60", "40", str(tone), str(tone), "120", "70", jvp, str(bpm)]))


def _write_stand_in_root(root):
    rows = [_row(1, 4, 0, "3", 70), _row(2, 6, 45, "Not Visible", 85, "N"), _row(3, 2, 90, "5", 60)]
    (root / "data").mkdir(parents=True)
    with (root / "dataset_info.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, INFO_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    for r in rows:
        d = root / "data" / r["Recording_Directory"]
        d.mkdir()
        fs = 2000.0
        p = TraceParams(duration_s=10.0, sample_rate_hz=fs, heart_rate_bpm=float(r["Average_BPM"]), hr_variability=0.0,
                        cvp_mean_mmhg=8.0, seed=1)
        tr = generate_trace(p)
        t = tr[:, 0]
        rr = 60.0 / p.heart_rate_bpm
        onsets = np.arange(-2.0, 12.0, rr)          # matches beat_onsets with zero variability
        ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in onsets) + 20 * rng.standard_normal(t.shape)
        with (d / "trace_data.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Time (s)", "CVP (mmHg)", "ECG (mV)"])
            for ti, ci, ei in zip(t, tr[:, 2], ecg):
                w.writerow([f"{ti:.5f}", f"{ci:.4f}", f"{ei:.3f}"])
        skin = np.array([180, 130, 100]) if r["Skin_Tone_Clinician"] != "6" else np.array([110, 75, 55])
        with MkvWriter(d / "K1_RGB.mkv", (64, 64), "rgb", 30.0) as w:
            for _ in range(15):
                frame = skin[None, None, :] + rng.standard_normal((64, 64, 3)) * 2.0
                frame[:8] = (40, 80, 160)       # a blue gown strip
                w.write(np.clip(frame, 0, 255).astype(np.uint8))
        if r["Recording_Directory"].endswith("_D"):
            with MkvWriter(d / "K1_IR.mkv", (64, 64), "gray16", 30.0) as w:
                for _ in range(15):
                    w.write(np.clip(3000 + rng.standard_normal((64, 64)) * 40, 0, 65535).astype(np.uint16))
    return rows


def test_quantiles_and_histogram():
    q = quantiles([1, 2, 3, 4, 5])
    assert q["p50"] == 3 and q["p05"] < q["p25"] < q["p75"] < q["p95"]
    h = histogram(["a", "b", "a"])
    assert h == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_population_priors_use_monk_and_posture_and_skip_not_visible(tmp_path):
    rows = _write_stand_in_root(tmp_path)
    pop = population_priors(rows)
    assert set(pop["monk_tone"]) == {"2", "4", "6"}
    assert set(pop["posture_deg"]) == {"0", "45", "90"}
    assert pop["depth_ir_share"] == pytest.approx(2 / 3)
    assert pop["jvp_height_cm"]["p50"] == 4.0                   # 3 and 5; "Not Visible" excluded
    assert pop["heart_rate_bpm"]["p50"] == 70


def test_detect_r_peaks_on_stand_in_ecg():
    fs = 2000.0
    t = np.arange(0, 10, 1 / fs)
    ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in np.arange(0.5, 10, 0.8))
    ecg += 20 * np.random.default_rng(0).standard_normal(t.shape)
    peaks = detect_r_peaks(ecg, fs)
    assert len(peaks) == 12 and abs(np.diff(peaks).mean() / fs - 0.8) < 0.01


@needs_ffmpeg
def test_waveform_priors_recover_cvp_shape(tmp_path):
    _write_stand_in_root(tmp_path)
    w = waveform_priors_for_recording(tmp_path / "data" / "P001_S01_R1_0_D" / "trace_data.csv")
    assert abs(w["heart_rate_bpm"] - 70) < 1 and w["hr_variability"] < 0.02
    assert 7 < w["cvp_mean_mmhg"] < 9
    assert w["a_wave_mmhg"] > 1.0 and w["v_wave_mmhg"] > 1.0 and w["x_descent_mmhg"] > 0.5
    assert 0.5 < w["resp_cvp_swing_mmhg"] < 3.0


def test_skin_mask_excludes_gown_and_black():
    rgb = np.zeros((20, 20, 3))
    rgb[:, :10] = (180, 130, 100)
    rgb[:, 10:] = (40, 80, 160)
    m = skin_mask(rgb, central=False)
    assert m[:, :10].all() and not m[:, 10:].any()


@needs_ffmpeg
def test_run_calibration_writes_schema_and_neckflix_config_validates(tmp_path):
    _write_stand_in_root(tmp_path)
    out = tmp_path / "priors.json"
    priors = run_calibration(tmp_path, out, n_recordings=3, seed=0)
    d = json.loads(out.read_text())
    assert d == priors
    assert set(d) == {"schema_version", "provenance", "population", "waveform", "appearance"}
    assert d["provenance"]["n_recordings"] == 3 and "Recording_Directory" not in json.dumps(d)
    assert "P001" not in json.dumps(d)
    assert 150 < d["appearance"]["skin_rgb_by_monk"]["4"][0] < 200
    assert 0.5 < d["appearance"]["read_noise_sd"]["p50"] < 4.0
    cfg = config_from_priors(out)
    validate(cfg)
    assert cfg.pulse.amplitude_levels.lo == 0.5 and cfg.pulse.amplitude_levels.hi == 2.0
    assert cfg.geometry.vein_visible_fraction.lo == 0.4
    assert cfg.appearance.monk_tone is not None and len(cfg.appearance.skin_rgb_by_monk) == 10
    assert cfg.streams.depth_ir_probability == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement calibrate.py**

```python
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
from .video import FORMATS

PRIORS_SCHEMA_VERSION = 1
QUANTILES = {"p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}


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
    sys_, dia = col("BP_Sys (mmHG)"), col("BP_Dia (mmHg)")
    pp = [s - d for s, d in zip(sys_, dia) if s > d]
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


def waveform_priors_for_recording(trace_csv: Path) -> dict:
    """CVP morphology and rhythm from one 'Time (s),CVP (mmHg),ECG (mV)' file."""
    data = np.loadtxt(trace_csv, delimiter=",", skiprows=1)
    t, cvp, ecg = data.T
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
    beat = np.mean(beats, axis=0)
    beat = beat - beat.mean()
    tb = (np.arange(len(beat)) - pre) / fs_d
    win = lambda lo, hi: beat[(tb >= lo) & (tb <= hi)]
    a, c = win(-0.20, 0.0).max(), win(0.02, 0.12).max()
    x, v, y = win(0.10, 0.30).min(), win(0.25, 0.50).max(), win(0.40, 0.65).min()

    resp = _bandpass(cvp_d, fs_d, 0.1, 0.5)
    return {
        "heart_rate_bpm": float(60.0 / rr.mean()),
        "hr_variability": float(rr.std() / rr.mean()),
        "cvp_mean_mmhg": float(cvp.mean()),
        "cvp_pulse_pressure_mmhg": float(np.percentile(cvp_d, 95) - np.percentile(cvp_d, 5)),
        "a_wave_mmhg": float(max(a, 0.0)),
        "c_wave_mmhg": float(max(c, 0.0)),
        "x_descent_mmhg": float(max(-x, 0.0)),
        "v_wave_mmhg": float(max(v, 0.0)),
        "y_descent_mmhg": float(max(-y, 0.0)),
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
    if len(xs) < 2:
        return float(np.sqrt(np.median(var))), 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(np.sqrt(max(intercept, 0.0))), float(max(slope, 0.0))


def appearance_priors_for_recording(rec_dir: Path, n_frames: int = 10, start_s: float = 10.0) -> dict:
    rgb = read_frames(rec_dir / "K1_RGB.mkv", "rgb", start_s, n_frames).astype(np.float64)
    if rgb.shape[0] < 2:
        rgb = read_frames(rec_dir / "K1_RGB.mkv", "rgb", 0.0, n_frames).astype(np.float64)
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
    pick = [usable[i] for i in sorted(rng.choice(len(usable), size=min(n_recordings, len(usable)), replace=False))]
    wave, look, monks = [], [], []
    for r in pick:
        d = root / "data" / r["Recording_Directory"]
        wave.append(waveform_priors_for_recording(d / "trace_data.csv"))
        look.append(appearance_priors_for_recording(d))
        monks.append(monk_of(r))
    priors = {
        "schema_version": PRIORS_SCHEMA_VERSION,
        "provenance": {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool_version": __version__,
                       "n_rows": len(rows), "n_recordings": len(pick), "seed": seed},
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: 6 passed. Two things that can bite: `np.loadtxt` on the stand-in CSV needs the header skipped (it is); and `read_frames` with `-ss 10` on a 15-frame (0.5 s) stand-in returns nothing, which is why `appearance_priors_for_recording` falls back to `start_s=0`.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/calibrate.py tests/test_calibrate.py
git commit -m "Add Neckflix calibration producing aggregated priors and the neckflix config mapping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Run calibration on the real drive and check in `priors/neckflix.json`

**Files:**
- Create: `priors/neckflix.json`
- Modify: `tests/test_presets.py` (add the checked-in-priors test)

**Interfaces:**
- Consumes: `run_calibration`, `neckflix()`.
- Produces: the committed priors file that makes `--preset neckflix` work on any machine.

This task needs the drive mounted at `/Volumes/Blue 4TB/CVP/Dataset/Neckflix`. If it is not mounted, stop and report; do not fabricate a priors file.

- [ ] **Step 1: Add the test that reads the checked-in file**

Append to `tests/test_presets.py`:
```python
def test_checked_in_neckflix_priors_load_and_validate():
    from synthetic_neck.presets import DEFAULT_PRIORS
    if not DEFAULT_PRIORS.exists():
        pytest.skip("priors/neckflix.json not generated yet")
    cfg = neckflix()
    validate(cfg)
    for seed in range(50):
        p = sample(cfg, np.random.default_rng(seed))
        assert p.geometry.vein_visible_fraction >= 0.4
        assert 0.5 <= p.pulse.amplitude_levels <= 2.0
        assert p.appearance.monk_tone in range(1, 11)
    text = DEFAULT_PRIORS.read_text()
    assert "Recording_Directory" not in text and "P0" not in text
```

- [ ] **Step 2: Run calibration**

```bash
uv run synthetic-neck calibrate --root "/Volumes/Blue 4TB/CVP/Dataset/Neckflix" --out priors/neckflix.json --n-recordings 30 --seed 0
```
Expected: `wrote priors/neckflix.json` in a few minutes (30 trace CSVs of ~20 MB and 20 short ffmpeg decodes). Then open the file and sanity-check: heart rate p50 near 80 bpm, `depth_ir_share` near 0.6, monk histogram covering tones 2–7, `skin_fraction` p50 above 0.1, `read_noise_sd` p50 between 0.5 and 4.

- [ ] **Step 3: Run the preset test and a smoke generation**

```bash
uv run pytest tests/test_presets.py -v
uv run synthetic-neck generate --preset neckflix --n 2 --out /tmp/neckflix_smoke --set video.frame_size=96
uv run synthetic-neck inspect --root /tmp/neckflix_smoke
```
Expected: tests pass; inspect prints artery and vein power ratios above 1 for both samples. If a ratio is below 1 the amplitude is under the calibrated noise floor at 96 px; try at the default 300 px before changing anything.

- [ ] **Step 4: Commit**

```bash
git add priors/neckflix.json tests/test_presets.py
git commit -m "Add calibrated Neckflix priors and the neckflix preset test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: README and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# synthetic-neck

Generates synthetic videos of a neck with propagating carotid (arterial) and
jugular (venous) pulsation, with full ground truth. Three presets:

| Preset | Use | Pulse amplitude |
|---|---|---|
| `lesson` | teaching: obvious signal, flat lighting | 7–12 levels |
| `benchmark` | signal-processing benchmarks with controlled degradation | 2–5 levels |
| `neckflix` | ranges calibrated from the Neckflix ICU dataset | 0.5–2 levels |

Every sample contains a visible arterial and venous pulse.

## Install

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on PATH.

    uv sync --all-extras

## Generate

    uv run synthetic-neck generate --preset lesson --n 5 --out data/synthetic_necks
    uv run synthetic-neck generate --preset neckflix --n 20 --jobs 4 --set pulse.amplitude_levels=1,1.5

`--set block.field=value` overrides any config field. Ranges are `lo,hi` or a
single fixed value; choices are `a|b|c`. Unknown names fail before rendering.

Each sample directory contains `trace.csv` (Time, ABP, CVP at 1 kHz),
`gray_video.mkv`, `rgbid_video.mkv` (streams `rgb`, and when present `ir`
and `depth` in 0.02 mm units), `vessel_ids.npy` (0 background, 1 artery,
2 vein) and `metadata.json` (effective config, drawn parameters, derived
quantities). The dataset root gets `dataset.json`.

## Inspect

    uv run synthetic-neck inspect --root data/synthetic_necks

Writes `fft_maps.png` per sample and prints artery/vein power at the heart
rate relative to background.

## Calibrate (needs the Neckflix drive)

    uv run synthetic-neck calibrate --root "/Volumes/Blue 4TB/CVP/Dataset/Neckflix" --out priors/neckflix.json

Writes aggregated quantiles and histograms only (no IDs, no pixels). The
checked-in `priors/neckflix.json` is what `--preset neckflix` reads.

## Develop

    uv run pytest

Design: `docs/superpowers/specs/2026-09-14-synthetic-neck-generator-design.md`.
```

- [ ] **Step 2: Full test run and a default-size generation**

```bash
uv run pytest -q
uv run synthetic-neck generate --preset lesson --n 1 --out /tmp/lesson_smoke
uv run synthetic-neck generate --preset benchmark --n 1 --out /tmp/benchmark_smoke
```
Expected: all tests pass; each generation finishes in under a couple of minutes and prints `wrote 1/1 samples`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document the synthetic-neck CLI, presets and outputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
