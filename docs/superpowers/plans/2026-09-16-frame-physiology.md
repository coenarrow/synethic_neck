# Physiology in the frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every skin pixel pulse with the stored PPG at neck timing and make the whole frame breathe with the stored RR trace, and guarantee the skin pulse is visible.

**Architecture:** Two new spatially uniform terms in the existing `PulseStage`: a skin PPG darkening (RGB and IR, no depth) driven by the stored finger PPG read `skin_lead_s` earlier, and a respiratory brightness gain plus depth offset driven by `x = 2·rr − 1`. The stage takes the sample's `TraceParams` for timing instead of a bare `site_delay_s`. The visibility retry loop gains a third SNR on the background mask.

**Tech Stack:** numpy only (this repository's idiom), zarr stores unchanged.

**Spec:** `docs/superpowers/specs/2026-09-16-frame-physiology-design.md`

## Global Constraints

- **No new tests, no pytest.** Never run `pytest`. The only check that may be run is remote-physiology's `tools/validate_cache.py` (Task 5 step 4 shows the command). Existing tests are edited only where this plan says, mechanically, and are not run.
- **numpy only** for array work; no einops, no torch in this repository.
- **Legacy is deleted, not adapted:** the `site_delay_s` keyword of `PulseStage` is removed, not kept as an alias.
- **Dependencies** go through `uv add` only; this plan adds none.
- **Exact values** from the spec: `skin_ratio = Range(0.1, 0.3)` (default 0.2), `resp_gain_frac = Range(0.003, 0.010)` (default 0.005), `resp_lift_mm = Range(0.5, 1.0)` (default 0.7), `skin_transit_s = Range(0.02, 0.04)` (default 0.03), `MIN_VISIBLE_SNR` stays 10 for all three regions.
- **Commits** end with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `feat/frame-physiology` in this repository (the synthetic_neck submodule checkout). Task 5 edits a file in the parent remote-physiology repository; it is the only task that leaves this repository.

---

### Task 1: Config fields

**Files:**
- Modify: `src/synthetic_neck/config.py` (`TraceConfig` ~line 126, `PulseConfig` ~line 179, `TraceParams` ~line 276, `PulseParams` ~line 361)

**Interfaces:**
- Produces: `TraceConfig.skin_transit_s`, `TraceParams.skin_transit_s`, `PulseConfig.skin_ratio / resp_gain_frac / resp_lift_mm`, `PulseParams.skin_ratio / resp_gain_frac / resp_lift_mm`, property `PulseParams.skin_amplitude_levels`. `sample()` picks the new fields up by name with no change.

- [ ] **Step 1: `TraceConfig`** — directly after the line `radial_to_finger_s: Range = Range(0.03, 0.08)` add:

```python
    skin_transit_s: Range = Range(0.02, 0.04)  # carotid -> neck skin capillaries (frame-physiology spec, section 3)
```

- [ ] **Step 2: `TraceParams`** — directly after the line `radial_to_finger_s: float = 0.05` add:

```python
    skin_transit_s: float = 0.03
```

- [ ] **Step 3: `PulseConfig`** — replace the class body so it reads:

```python
class PulseConfig:
    amplitude_levels: Range = Range(7, 12)     # artery, green channel, peak-to-peak
    vein_ratio: Range = Range(0.4, 0.6)        # vein amplitude / artery amplitude
    skin_ratio: Range = Range(0.1, 0.3)        # skin-wide PPG amplitude / artery amplitude
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: Range = Range(0.35, 0.35)
    artery_lift_mm: Range = Range(0.3, 0.3)
    vein_lift_mm: Range = Range(0.5, 0.5)
    resp_gain_frac: Range = Range(0.003, 0.010)   # whole-scene brightness swing with breathing, fractional
    resp_lift_mm: Range = Range(0.5, 1.0)         # whole-frame depth swing with breathing, mm
```

- [ ] **Step 4: `PulseParams`** — replace the class body so it reads:

```python
class PulseParams:
    amplitude_levels: float = 10.0
    vein_ratio: float = 0.5
    skin_ratio: float = 0.2
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: float = 0.35
    artery_lift_mm: float = 0.3
    vein_lift_mm: float = 0.5
    resp_gain_frac: float = 0.005
    resp_lift_mm: float = 0.7

    @property
    def vein_amplitude_levels(self) -> float:
        return self.amplitude_levels * self.vein_ratio

    @property
    def skin_amplitude_levels(self) -> float:
        return self.amplitude_levels * self.skin_ratio
```

- [ ] **Step 5: Check it imports and samples**

Run: `uv run python -c "from synthetic_neck.config import GeneratorConfig, sample; import numpy as np; p = sample(GeneratorConfig(), np.random.default_rng(1)); print(p.pulse.skin_amplitude_levels, p.pulse.resp_gain_frac, p.pulse.resp_lift_mm, p.trace.skin_transit_s)"`
Expected: four numbers, the first between 0.7 and 3.6, the last between 0.02 and 0.04.

- [ ] **Step 6: Commit**

```bash
git add src/synthetic_neck/config.py
git commit -m "Add skin PPG and respiration render ranges

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Skin PPG and respiration in the pulse stage

**Files:**
- Modify: `src/synthetic_neck/render/pulse.py` (whole file)
- Modify: `src/synthetic_neck/render/renderer.py` (`PulseStage(...)` call, `frame`, `ir_frame`)
- Modify: `tests/test_render_pulse.py` (`_stage` fixture only; mechanical; do not run)

**Interfaces:**
- Consumes: Task 1's fields and `TraceParams.abp_site_delay_s`, `TraceParams.pep_s`, `TraceParams.r_to_ppg_foot_s` (existing properties).
- Produces: `PulseStage.__init__(pulse, gp, geometry, pixel_scale_mm, trace, timing: TraceParams)`; attributes `entry_delay_art_s`, `r_to_skin_foot_s`, `skin_lead_s` (floats); methods `skin_drive(time_s) -> float`, `resp_excursion(time_s) -> float`, `resp_gain(time_s) -> float`; `depth_lift_mm` now includes the respiratory offset.

- [ ] **Step 1: Rewrite `src/synthetic_neck/render/pulse.py`** to exactly:

```python
"""Pulse stage: physiology -> additive pixel modulation.

The two vessel pressures propagate along their vessels with per-pixel delays. Two further terms are spatially
uniform: the skin-wide PPG (the stored finger PPG read at neck timing) darkens every pixel of the cylinder, and
breathing brightens the whole scene and lifts the whole frame. Design:
docs/superpowers/specs/2026-09-16-frame-physiology-design.md.
"""
from __future__ import annotations

import numpy as np

from ..config import GeometryParams, PulseParams, TraceParams
from ..geometry import (VesselGeometry, arterial_pwv_m_s, delay_map, tube_fields, venous_pwv_m_s,
                        visibility_taper)


def normalise(x: np.ndarray) -> np.ndarray:
    """Map a trace to [-0.5, 0.5] using its 1st/99th percentiles (robust to noise)."""
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo), 0, 1) - 0.5


class PulseStage:
    def __init__(self, pulse: PulseParams, gp: GeometryParams, geometry: VesselGeometry,
                 pixel_scale_mm: float, trace: np.ndarray, timing: TraceParams):
        self.p = pulse
        self.t = trace[:, 0]
        self.abp_n = normalise(trace[:, 1])
        self.cvp_n = normalise(trace[:, 2])
        self.ppg_n = normalise(trace[:, 4])
        self.rr = trace[:, 5]
        # The stored ABP is measured at an arm site `site_delay_s` after the aortic root; the carotid pulse is
        # the same waveform that much earlier, before its own root -> neck propagation (delay_art).
        self.site_delay_s = float(timing.abp_site_delay_s)
        self.map_art = float(trace[:, 1].mean())
        self.mean_cvp = float(trace[:, 2].mean())
        self.pwv_art = arterial_pwv_m_s(self.map_art)
        self.pwv_vein = venous_pwv_m_s(self.mean_cvp)

        self.w_art, s_art = tube_fields(geometry, "artery")
        w_vein, s_vein = tube_fields(geometry, "vein")
        self.w_vein = w_vein * visibility_taper(s_vein, geometry.length_px, geometry.vein_visible_fraction)
        self.delay_art = delay_map(s_art, self.pwv_art, gp.heart_to_neck_artery_m, pixel_scale_mm)
        self.delay_vein = delay_map(s_vein, self.pwv_vein, gp.heart_to_neck_vein_m, pixel_scale_mm)
        # The stored PPG is a finger pulse. The neck skin fills after the carotid's entry delay plus a capillary
        # transit, earlier than the finger, so its drive is the stored PPG read `skin_lead_s` ahead of frame time.
        self.entry_delay_art_s = float(gp.heart_to_neck_artery_m / self.pwv_art)
        self.r_to_skin_foot_s = float(timing.pep_s + self.entry_delay_art_s + timing.skin_transit_s)
        self.skin_lead_s = float(timing.r_to_ppg_foot_s - self.r_to_skin_foot_s)
        self._gain = np.asarray(pulse.channel_gain, dtype=np.float64)

    def pulse_maps(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised pressure at every pixel of each vessel at `time_s`."""
        p_art = np.interp(time_s - self.delay_art + self.site_delay_s, self.t, self.abp_n)
        p_vein = np.interp(time_s - self.delay_vein, self.t, self.cvp_n)
        return p_art, p_vein

    def skin_drive(self, time_s: float) -> float:
        """Normalised PPG of the neck skin at `time_s`, uniform over the frame."""
        return float(np.interp(time_s + self.skin_lead_s, self.t, self.ppg_n))

    def resp_excursion(self, time_s: float) -> float:
        """Chest excursion in [-1, 1] at `time_s`; +1 at end-inspiration. No lag: the neck moves with the chest."""
        return float(2.0 * np.interp(time_s, self.t, self.rr) - 1.0)

    def resp_gain(self, time_s: float) -> float:
        """Whole-scene brightness factor: the neck tilts towards the light on inspiration."""
        return 1.0 + self.p.resp_gain_frac * self.resp_excursion(time_s)

    def _green_mod(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        # More blood volume (higher pressure in a vessel, a fuller capillary bed) -> more absorption -> darker.
        return -(self.p.amplitude_levels * self.w_art * p_art
                 + self.p.vein_amplitude_levels * self.w_vein * p_vein
                 + self.p.skin_amplitude_levels * self.skin_drive(time_s))

    def rgb_mod(self, time_s: float) -> np.ndarray:
        return self._green_mod(time_s)[..., None] * self._gain[None, None, :]

    def ir_mod(self, time_s: float) -> np.ndarray:
        return self.p.ir_gain * self._green_mod(time_s)

    def depth_lift_mm(self, time_s: float) -> np.ndarray:
        """Lift towards the camera in mm: the vessel pulses, plus the whole neck rising on inspiration."""
        p_art, p_vein = self.pulse_maps(time_s)
        return (self.p.artery_lift_mm * self.w_art * p_art + self.p.vein_lift_mm * self.w_vein * p_vein
                + self.p.resp_lift_mm * self.resp_excursion(time_s))
```

- [ ] **Step 2: Wire the renderer.** In `src/synthetic_neck/render/renderer.py`, replace the `PulseStage(...)` construction with:

```python
        self.pulse = PulseStage(params.pulse, params.geometry, g, self.pixel_scale_mm, trace, params.trace)
```

and replace `frame` and `ir_frame` with:

```python
    def frame(self, i: int, noise: bool = True) -> np.ndarray:
        """RGB uint8 (H, W, 3) at output resolution."""
        t = self.frame_time(i)
        img = (self.base.rgb + self.pulse.rgb_mod(t)) * self.pulse.resp_gain(t)
        img = self.illumination(img, i)
        return self.sensor.rgb(img, noise=noise)

    def ir_frame(self, i: int, noise: bool = True) -> np.ndarray:
        """Near-infrared uint8 (H, W)."""
        t = self.frame_time(i)
        img = (self.base.ir + self.pulse.ir_mod(t)) * self.pulse.resp_gain(t)
        img = self.illumination(img, i)
        return self.sensor.ir(img, noise=noise)
```

`depth_frame` is unchanged: the respiratory offset arrives through `depth_lift_mm`.

- [ ] **Step 3: Mechanical fixture edit.** In `tests/test_render_pulse.py`, replace `_stage` with:

```python
def _stage(**geom):
    tp = TraceParams(heart_rate_bpm=60, hr_variability=0.0)
    tr = generate_trace(tp)
    g = VesselGeometry(frame_size=120, angle_deg=90, length_px=80, centre_xy=(60, 60), **geom)
    return PulseStage(PulseParams(amplitude_levels=10.0, vein_ratio=0.5), GeometryParams(), g, 0.5, tr, tp), g, tr
```

Do not run the test.

- [ ] **Step 4: Smoke the renderer** (a run of the tool, not a test):

Run:
```bash
uv run python - <<'PY'
import numpy as np
from synthetic_neck.config import GeneratorConfig, sample
from synthetic_neck.render.renderer import Renderer
from synthetic_neck.traces import generate_trace
from synthetic_neck.config import apply_override
cfg = apply_override(GeneratorConfig(), "video.frame_size", "48")
p = sample(cfg, np.random.default_rng(3))
r = Renderer(p, generate_trace(p.trace))
print("lead", r.pulse.skin_lead_s, "skin foot", r.pulse.r_to_skin_foot_s, "finger foot", p.trace.r_to_ppg_foot_s)
bg = r.ids == 0
g = np.array([r.frame(i, noise=False)[..., 1][bg].mean() for i in range(60)])
d = np.array([r.depth_frame(i, noise=False).mean() for i in range(60)])
print("skin green ptp", np.ptp(g), "depth mean ptp mm", np.ptp(d))
PY
```
Expected: `lead` between 0.01 and 0.16 and `skin foot` < `finger foot`; skin green ptp of at least about 0.5 level; depth mean ptp of at least about 0.3 mm over 2 s.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_neck/render/pulse.py src/synthetic_neck/render/renderer.py tests/test_render_pulse.py
git commit -m "Pulse the whole skin with the PPG and breathe the frame with rr

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Skin visibility gate, metadata and inspect

**Files:**
- Modify: `src/synthetic_neck/generate.py` (`_render`, `generate_sample`)
- Modify: `src/synthetic_neck/inspect.py` (`vessel_snr`, `inspect_root`, `vessel_power_ratio` docstring)

**Interfaces:**
- Consumes: `PulseStage.r_to_skin_foot_s`, `PulseStage.skin_lead_s` from Task 2.
- Produces: `_render(...) -> (Renderer, list[str], float, float, float)`; `vessel_snr(...) -> dict[str, tuple[float, float, float]]` (artery, vein, skin); metadata keys `derived.r_to_skin_foot_s`, `derived.skin_lead_s`, `visibility.skin_snr`.

- [ ] **Step 1: `_render`** in `generate.py` becomes:

```python
def _render(params, sink) -> tuple[Renderer, list[str], float, float, float]:
    """Render every stream in one pass into `sink`; return (renderer, streams, green artery SNR, green vein SNR,
    green skin SNR), the skin being the background mask."""
    has_di = params.streams.has_depth_ir
    streams = ["rgb", "ir", "depth"] if has_di else ["rgb"]
    fps = params.video.fps
    # The sink decides which trace the video is rendered from (FolderSink: the CSV read back off disk).
    trace = sink.begin(generate_trace(params.trace), streams, params.video.n_frames, params.video.frame_size, fps)
    r = Renderer(params, trace)
    masks = (r.ids == 1, r.ids == 2, r.ids == 0)                 # artery, vein, skin
    means = np.empty((3, r.n_frames))
    for i in range(r.n_frames):
        rgb = r.frame(i)
        g = rgb[..., 1].astype(np.float64)
        for k, mask in enumerate(masks):
            means[k, i] = g[mask].mean()
        # Arguments evaluate left to right, keeping the sensor noise draws in rgb -> ir -> depth order.
        sink.frame(rgb, r.ir_frame(i) if has_di else None, r.depth_frame(i) if has_di else None)
    hr_hz = params.trace.heart_rate_bpm / 60.0
    artery_snr, vein_snr, skin_snr = (cardiac_snr(m, fps, hr_hz) for m in means)
    return r, streams, artery_snr, vein_snr, skin_snr
```

- [ ] **Step 2: the gate** in `generate_sample`. Update the docstring sentence to "Guarantees a testable green-channel pulse: both vessels and the skin are checked and, if any is under MIN_VISIBLE_SNR, the sample is redrawn ..." and replace the attempt loop with:

```python
        last = (0.0, 0.0, 0.0)
        for attempt in range(MAX_VISIBILITY_ATTEMPTS):
            params = sample(config, _attempt_rng(seed, attempt))
            r, streams, artery_snr, vein_snr, skin_snr = _render(params, sink)
            last = (artery_snr, vein_snr, skin_snr)
            if min(last) >= MIN_VISIBLE_SNR:
                break
        else:
            raise RuntimeError(f"seed {seed}: no draw reached green SNR {MIN_VISIBLE_SNR} in every region in "
                               f"{MAX_VISIBILITY_ATTEMPTS} attempts (last artery {last[0]:.1f}, vein {last[1]:.1f}, "
                               f"skin {last[2]:.1f})")
```

The message must keep the words `no draw reached` (an existing test matches on them).

- [ ] **Step 3: metadata.** In the `meta` dict, after the `"r_to_ppg_foot_s"` line of `derived` add:

```python
                "r_to_skin_foot_s": pulse.r_to_skin_foot_s,
                "skin_lead_s": pulse.skin_lead_s,
```

and change the `visibility` entry to:

```python
            "visibility": {"channel": "G", "min_snr": MIN_VISIBLE_SNR, "attempts": attempt + 1,
                           "artery_snr": artery_snr, "vein_snr": vein_snr, "skin_snr": skin_snr},
```

- [ ] **Step 4: `inspect.py`.** Replace `vessel_snr` with:

```python
def vessel_snr(sample_dir: Path) -> dict[str, tuple[float, float, float]]:
    """(artery, vein, skin) cardiac SNR per channel, from each region mask's per-frame mean signal; the skin is
    the background mask."""
    meta = json.loads((sample_dir / "metadata.json").read_text())
    fps, hr_hz = meta["fps"], meta["params"]["trace"]["heart_rate_bpm"] / 60.0
    ids = np.load(sample_dir / "vessel_ids.npy")
    out = {}
    for name, v in load_channels(sample_dir).items():
        out[name] = tuple(cardiac_snr(v[:, ids == k].mean(1), fps, hr_hz) for k in (1, 2, 0))
    return out
```

Change `vessel_power_ratio`'s docstring to `"""(artery/skin, vein/skin) mean power ratios per channel: how far each vessel stands out above the pulsing skin around it."""`. In `inspect_root`, replace the two lines from `r = vessel_power_ratio(...)` to the end of the `print` with:

```python
        r = vessel_power_ratio(power, ids)["G"]
        skin = vessel_snr(d)["G"][2]
        print(f"{name}: {out.name}  G power artery x{r[0]:.1f} vein x{r[1]:.1f} over skin  skin SNR {skin:.0f}  "
              f"phase artery {s[0]:+.2f} vein {s[1]:+.2f} diff {s[2]:+.2f} rad")
```

- [ ] **Step 5: Run the generator** (folder mode needs ffmpeg; if `ffmpeg` is missing use `--zarr` and skip the inspect line):

```bash
uv run synthetic-neck generate --preset lesson --n 2 --set video.frame_size=64 --out /tmp/sn_frames_t3
uv run synthetic-neck inspect --root /tmp/sn_frames_t3
uv run python -c "import json; m=json.load(open('/tmp/sn_frames_t3/1/metadata.json')); print(m['visibility'], m['derived']['r_to_skin_foot_s'], m['derived']['skin_lead_s'])"
```
Expected: two samples, each inspect line ends with a skin SNR and the phases; `visibility` has `skin_snr >= 10`.

- [ ] **Step 6: Commit**

```bash
git add src/synthetic_neck/generate.py src/synthetic_neck/inspect.py
git commit -m "Guarantee a visible skin PPG and report it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: README

**Files:**
- Modify: `README.md` (`## Traces` section end, `## Inspect` section)

- [ ] **Step 1:** Replace the last sentence of the Traces section's closing paragraph, `The carotid pixels are rendered from ABP shifted back to central timing.`, with `The carotid pixels are rendered from ABP shifted back to central timing (below).`, then append after that paragraph:

```markdown
### What the frames carry

Every signal a model is asked to predict has a footprint in the video, in
the form a real neck video would show it
(`docs/superpowers/specs/2026-09-16-frame-physiology-design.md`):

| Signal | Region | Mechanism |
| --- | --- | --- |
| ABP | carotid mask | ABP shifted back to central timing, propagated along the vessel; darkening in RGB and IR, a depth lift |
| CVP | jugular mask | CVP propagated along the vessel; darkening in RGB and IR, a depth lift |
| PPG | every skin pixel, vessels included | the stored finger PPG read `skin_lead_s` earlier, so the neck skin fills after the carotid (`pep_s` + the artery's entry delay + `skin_transit_s`) and before the finger; uniform darkening in RGB and IR at `skin_ratio` times the artery amplitude, no depth lift |
| RR | whole frame | brighter by `resp_gain_frac` and nearer by `resp_lift_mm` at end-inspiration, no lag |
| ECG | everywhere | rate only: every term sits on the one cardiac timeline |

The visibility guarantee covers all three regions: a sample is redrawn until
the green-channel cardiac SNR of the artery, the vein and the skin (the
background mask) are each at least 10; `metadata.json` records the three
under `visibility`.
```

- [ ] **Step 2:** Replace the Inspect section's sentence `Writes \`fft_maps.png\` per sample and prints artery/vein power at the heart rate relative to background.` with:

```markdown
Writes `fft_maps.png` per sample and prints artery/vein power at the heart
rate relative to the surrounding skin, which pulses with the PPG, and the
skin's own cardiac SNR.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: what the frames carry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: remote-physiology dataset doc

**Files:**
- Modify: `../../../dataset/data_loader/SYNTHETIC_NECK.md` (the parent repository; bullets under "What the generator writes")

This task's commit goes in the parent repository, on a branch `feat/synthetic-neck-frames` created from its `main`. The submodule pointer bump happens later, at finishing, not here.

- [ ] **Step 1:** Replace the `**Frames**` bullet with:

```markdown
- **Frames**: rendered square at `video.frame_size` (300 by default), with
  no resize at write time; resizing is consumer-side as usual. Chunks are
  `(C, min(32, T), H, W)`. Every stored signal has a footprint: the carotid
  and jugular masks carry the delayed ABP and CVP (darkening in RGB and IR,
  a depth lift); every skin pixel carries the PPG at neck timing (a uniform
  darkening, weaker than the carotid, no depth lift); the whole frame
  brightens and moves nearer with `rr`; ECG is present as rate only. The
  submodule's `docs/superpowers/specs/2026-09-16-frame-physiology-design.md`
  gives the terms.
```

- [ ] **Step 2:** In the `**vessel_ids**` bullet change `(nothing in the scene moves)` to `(nothing in the scene moves in-plane)`.

- [ ] **Step 3:** Replace the `**Guaranteed pulse**` bullet with:

```markdown
- **Guaranteed pulse**: the generator redraws a sample, up to 20 times,
  until the green channel's region-averaged cardiac SNR is at least 10 for
  the artery, the vein and the skin. IR and depth carry no such guarantee.
```

- [ ] **Step 4: Validate a fresh cache** (the only test permitted). From the parent repository root:

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset lesson --n 4 --set video.frame_size=64 \
    --set streams.depth_ir_probability=0.5 --out /tmp/sn_frames_zarr
uv run python tools/validate_cache.py /tmp/sn_frames_zarr
```
Expected: `4/4 stores pass`.

- [ ] **Step 5: Commit** in the parent repository:

```bash
git add dataset/data_loader/SYNTHETIC_NECK.md
git commit -m "docs: synthetic-neck frames carry the skin PPG and respiration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
