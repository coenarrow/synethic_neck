# Synthetic neck video generator — design

Date: 2026-09-14
Status: approved in discussion, awaiting written review

## 1. Purpose

A standalone Python package and CLI that generates synthetic videos of a human
neck showing propagating arterial (carotid) and jugular venous pulses, with full
ground truth. One generator serves three uses, selected by preset and flags:

| Use | Preset | Character |
|---|---|---|
| Teaching dataset for pytorch_lessons | `lesson` | clean, large pulse amplitude, no illumination effects |
| Benchmark for rPPG/JVP signal-processing methods | `benchmark` | moderate amplitude, controlled degradations |
| Pretraining / augmentation for models evaluated on Neckflix | `neckflix` | ranges calibrated from the real Neckflix dataset |

Every sample must contain a visible arterial **and** venous pulse above the
noise floor. "JVP not visible" cases are deliberately never generated.

## 2. Relationship to existing code

The starting point is the uncommitted `src/synthetic/` tree in
`/Users/20759193/repos/pytorch_lessons` (traces, geometry, render, video,
generate, inspect, and 13 tests). It is copied into this repo and diverges from
here; pytorch_lessons is not a consumer or dependency.

The existing output layout is kept unchanged (section 7).

## 3. Use of the Neckflix dataset

Neckflix (`/Volumes/Blue 4TB/CVP/Dataset/Neckflix`, 332 recordings, 50 ICU/CCU
patients, two Azure Kinects, 650×650 ~30 fps 30 s RGB/IR/Depth, 20 kHz ECG and
CVP) is used **only** as an offline calibration source:

- **Parameter priors** from `dataset_info.csv` and trace files.
- **Appearance statistics** measured from a sample of frames.

No real pixels and no real traces are used in generation or shipped. The
calibration output (`priors/neckflix.json`) contains aggregated quantiles and
histograms only, no recording IDs or per-patient rows.

Dataset facts that matter for calibration (verified with the dataset owner):

- Folder suffix `_D` / `_N` = depth+IR recorded / not recorded. All recordings
  are indoors; there is no day/night distinction.
- `Skin_Tone_*` columns are Monk Skin Tone scores (1–10).
- The `0/45/90` in folder names is the participant's posture (head elevation),
  not a camera property.
- `JVP Height Estimate (cm)` contains `Not Visible` for 31 rows; these rows are
  excluded from JVP-related priors.

## 4. Package layout

```
pyproject.toml                    uv project, python >=3.13
src/synthetic_neck/
  config.py       Range, GeneratorConfig blocks, sample() -> SampleParams
  presets.py      lesson(), benchmark(), neckflix() -> GeneratorConfig
  traces.py       ABP/CVP synthesis from concrete TraceParams
  geometry.py     vessel layout, camera model, propagation delays
  render/
    base.py       static scene: skin, texture, vignette, cylindrical shading
    pulse.py      pressure-to-pixel modulation with propagation delay
    illumination.py  ambient gain, drift, flicker, specular highlight
    camera.py     sensor noise, blur, quantisation
    renderer.py   composes stages into frame(i), ir_frame(i), depth_frame(i)
  video.py        MkvWriter, mux, read_mkv (ffmpeg pipes)
  generate.py     generate_sample(), dataset loop, parallel jobs
  inspect.py      FFT QA maps
  calibrate.py    Neckflix -> priors/neckflix.json (offline)
  cli.py          `synthetic-neck generate | inspect | calibrate`
priors/neckflix.json              checked in
tests/
docs/superpowers/specs/
```

Runtime dependencies: `numpy`; `ffmpeg`/`ffprobe` on PATH. Optional extra
`calibrate`: `pandas`, `h5py`. Dev: `pytest`.

## 5. Configuration model

### Range

```python
@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    def draw(self, rng) -> float
```

A fixed value is `Range(v, v)`. Integer-valued fields use `IntRange`.
Categorical fields use `Choice(values, weights)`.

### GeneratorConfig

One dataclass composed of blocks. Every random quantity is a `Range`,
`IntRange` or `Choice`; non-random settings are plain scalars/tuples.

| Block | Contents |
|---|---|
| `video` | `frame_size`, `fps`, `duration_s` (scalars) |
| `streams` | `depth_ir_probability` (probability a sample includes IR+Depth streams) |
| `trace` | HR, HRV, respiratory rate, systolic/diastolic, CVP mean, CVP wave amplitudes (a, c, x, v, y), upstroke delay, respiratory swing, transducer noise, `posture_deg` (Choice) |
| `geometry` | vessel lengths, separation, widths (mm), angle, centre jitter, artery side, `vein_visible_fraction` |
| `camera` | distance, native width, HFOV, crop, output size |
| `appearance` | skin base colour or Monk score + mapping, texture sd, vignette, static vessel contrast, IR base, neck radius |
| `pulse` | `amplitude_levels` (artery), `vein_ratio`, channel gains, IR gain, depth lift |
| `illumination` | `ambient_gain`, `drift_sd`, `drift_tau_s`, `flicker_amp`, `flicker_hz`, `specular_amp`, `specular_sigma` |
| `sensor` | `read_noise_sd`, `shot_noise_gain`, `blur_sigma_px`, `depth_noise_mm_at_1m`, quantisation |
| `motion` | reserved, empty (deferred) |
| `rhythm` | reserved, empty (deferred; arrhythmia) |

### Sampling

`sample(config, rng) -> SampleParams` walks blocks in a fixed order
(video, streams, trace, camera, geometry, appearance, pulse, illumination,
sensor) and returns a parallel set of concrete frozen dataclasses. Stages
consume only `SampleParams`. Seed `base_seed + i` for sample `i`.

### Posture and vein visibility

`posture_deg` sets the hydrostatic column between right atrium and neck. From
posture and sampled CVP mean, the generator derives `vein_visible_fraction`,
the fraction of the vein's length (from the caudal end) whose weight map is
kept; the rest tapers to zero. It is clamped to the block's `Range`
(`lesson`: 1.0–1.0; `benchmark`: 0.6–1.0; `neckflix`: 0.4–1.0), so a venous
signal is always present.

### Presets and overrides

Presets are functions returning a fully populated `GeneratorConfig`.
`neckflix()` loads `priors/neckflix.json` and maps p05–p95 quantiles to
`Range` fields; a missing file raises an error naming `calibrate`.

CLI overrides use dotted names: `--set pulse.amplitude_levels=0.5,2`,
`--set illumination.flicker_amp=0`. Names and types are validated against the
dataclasses before any rendering.

Target amplitudes (8-bit levels, artery): `lesson` 7–12, `benchmark` 2–5,
`neckflix` 0.5–2.

## 6. Renderer stages

Per frame, per channel: `base scene → pulse → illumination → sensor`. Each
stage is a small class with `__call__(img, t) -> img` reading only its own
params block. Identity parameter values disable a stage. Frames are rendered at
crop resolution (e.g. 650) and area-downsampled to output size (e.g. 300) after
blur and before noise/quantisation, as today.

- **Base scene**: skin colour (from Monk score mapping in `neckflix`), static
  texture, radial vignette, faint static vessel darkening, and **cylindrical
  shading** in RGB and IR using the same `neck_radius_mm` as the depth channel
  (Lambertian cosine falloff from the camera axis).
- **Pulse**: existing per-pixel propagation-delay modulation. Artery amplitude
  from `pulse.amplitude_levels`; vein amplitude = artery × `vein_ratio`, then
  multiplied by the vein visibility taper. IR and depth use their own gains.
- **Illumination**: multiplicative gain
  `ambient_gain × (1 + drift(t)) × (1 + flicker(t))` plus an additive specular
  Gaussian blob at the shading peak. Drift is an Ornstein–Uhlenbeck process
  with `drift_tau_s`; flicker is a sinusoid at `flicker_hz` sampled at frame
  times (aliasing is intended).
- **Sensor**: Gaussian blur (`blur_sigma_px`, at crop resolution), area
  downsample, signal-dependent noise `sd = sqrt(read² + shot_gain × signal)`,
  then 8-bit quantisation (uint8) for RGB/IR. Depth: noise sd grows with
  distance (`depth_noise_mm_at_1m × (d/1000)²`), then 0.02 mm uint16 units.

Deferred (blocks reserved): subject motion, breathing displacement,
swallowing, arrhythmia, oblique projection, clutter/occlusion.

## 7. Outputs

Per sample directory `<out>/<i>/` (unchanged layout):

| File | Content |
|---|---|
| `trace.csv` | `Time,ABP,CVP` at 1000 Hz |
| `gray_video.mkv` | FFV1, 8-bit BT.601 luma |
| `rgbid_video.mkv` | FFV1, streams titled `rgb` (bgr0), and when present `ir` (gray8), `depth` (gray16le, 0.02 mm) |
| `vessel_ids.npy` | uint8 (H,W): 0 background, 1 artery, 2 vein |
| `metadata.json` | `seed`, preset name, effective `GeneratorConfig`, drawn `SampleParams`, derived quantities (pixel scale, PWVs, delay ranges), stream list, label legend |

Dataset root gets `dataset.json`: preset, overrides, base seed, n, package
version, git commit, timestamp.

## 8. Generation loop and error handling

`generate_sample(config, seed, out_dir)`: sample → trace → write CSV →
renderer → single pass writing all streams → mux → mask → metadata.

`generate` CLI: `--preset`, `--set` (repeatable), `--n`, `--start`,
`--seed`, `--out`, `--jobs`. Samples run in a multiprocessing pool, one
sample per process.

Fail fast before rendering: unknown `--set` field, wrong type, `lo > hi`,
geometry that cannot fit in frame, `vein_visible_fraction` lower bound < 0.1,
missing priors for `neckflix`, ffmpeg/ffprobe missing. A sample that fails
mid-render deletes its partial directory; the batch continues and reports
failed seeds at the end with non-zero exit status.

## 9. Calibration (`synthetic-neck calibrate`)

Inputs: `--root <Neckflix dir>`, `--n-recordings` (default 30), `--seed`,
`--out priors/neckflix.json`.

1. **Population priors** from `dataset_info.csv`: quantiles of `Average_BPM`,
   age, neck circumferences, BP; histograms of sex, Monk skin tone
   (clinician column, fallback recorder), posture (from folder name), depth/IR
   availability share, JVP height (excluding `Not Visible`), joint
   posture × JVP-height table.
2. **Waveform priors** from `trace_data.csv` of the sampled recordings:
   R-peaks from ECG (simple bandpass + peak pick), beat-aligned CVP average
   per recording → CVP mean, pulse pressure, relative a/c/v amplitudes and
   x/y depths, respiratory swing (band 0.1–0.5 Hz), HRV (RR sd / mean).
3. **Appearance priors** from ~10 RGB and IR frames per sampled recording,
   decoded with ffmpeg: skin mask = central-region pixels passing a
   colour/brightness threshold (fraction selected is reported); mean skin
   colour per Monk class; static texture sd (spatial high-pass); temporal
   noise in a still background region fitted as read + shot terms; IR base
   level; ambient gain spread after removing the per-Monk mean.

Output JSON sections: `population`, `waveform`, `appearance`, `provenance`
(date, n recordings, seed, tool version). Each numeric prior is
`{p05, p25, p50, p75, p95}`; categoricals are `{value: weight}`.

## 10. Testing

Port the 13 existing tests. Add:

- `Range`/`Choice` drawing, config round-trip through JSON, `--set` parsing
  and validation errors.
- Each preset samples 50 seeds without validation errors.
- Stage identity: illumination and sensor stages with identity params leave
  the image unchanged.
- Cylindrical shading peak coincides with the depth minimum.
- Measured noise sd matches configured sensor params within tolerance.
- Vein taper: `vein_visible_fraction` respected and never below the bound.
- Every preset: FFT QA detects artery and vein power above background and an
  artery–vein phase difference, at that preset's amplitude.
- Calibrate: unit tests on synthetic stand-in `dataset_info.csv`, traces and
  frames (CI has no drive); output schema check.

## 11. Out of scope for this spec

Motion, breathing displacement, swallowing, arrhythmia, oblique camera
projection, clutter, Neckflix-compatible output layout, and any use of real
pixels or traces in generated samples.
