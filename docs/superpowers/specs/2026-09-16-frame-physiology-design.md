# Physiology in the frames: skin PPG and respiration — design

Date: 2026-09-16
Status: approved in discussion

## 1. Purpose

After the traces change (`2026-09-16-physiological-traces-design.md`) every
sample stores ABP, CVP, ECG, PPG and RR, but the frames carry only the two
vessel pulses. A model trained on the synthetic cache could therefore learn
ABP and CVP from the video but not PPG or respiration, which is the point of
the dataset: every signal remote-physiology predicts must be recoverable from
the frames, in the form a real neck video would show it.

This change makes the whole skin cylinder pulse with the PPG and the whole
frame breathe with the RR trace, and records the guarantee that the skin
pulse is visible. ECG is not rendered: only its rate is recoverable from a
video, and that is already shared with every other term through the one
cardiac timeline.

What the frames carry after this change:

| Signal | Region | Mechanism | Status |
| --- | --- | --- | --- |
| ABP | carotid mask | delayed ABP darkening and depth lift | unchanged |
| CVP | jugular mask | delayed CVP darkening and depth lift | unchanged |
| PPG | every skin pixel, vessels included | uniform darkening driven by the stored PPG at neck timing, RGB and IR only | **new** |
| RR | whole frame | a multiplicative brightness gain and a whole-frame depth offset, both driven by the stored `rr` | **new** |
| ECG | everywhere | rate only, from the shared cardiac timeline | unchanged, documented |

## 2. Decisions

| Question | Decision |
| --- | --- |
| Skin PPG amplitude | **A ratio of the artery amplitude**, `PulseConfig.skin_ratio`, `Range(0.1, 0.3)`, beside `vein_ratio`. The carotid is stronger than the skin by construction and a `--set` on `amplitude_levels` moves both. Rejected: absolute levels per preset (a preset could invert the order), and a fractional AC/DC of the pixel level (under one level on `benchmark`). |
| Skin PPG waveform | **The stored finger PPG**, normalised like the vessel drives and shifted **earlier** to neck timing (section 3). The pixels stay a pure function of the stored ground truth, as for the carotid. The neck skin and the finger share one morphology; a distinct neck waveform stays out of scope, as the traces spec left it. |
| Skin PPG channels | RGB with the existing `channel_gain`, IR with the existing `ir_gain`: the same chromophore as the vessels. **No depth lift**: capillary volume does not move the skin surface measurably. |
| Respiration in the frames | **A brightness term and a depth term, no pixel motion.** Breathing tilts the neck against the light and raises it towards the camera. Rejected for this pass: sub-pixel scene motion (needs per-frame resampling of the base scene, the illumination field and the vessel masks) and traces-only (RR would be recoverable only through the few percent riding on the PPG and the vessel swings). |
| Respiratory sign | `x = 2·rr − 1`, so `+1` at end-inspiration: **brighter and nearer on inspiration**. No sign switch: the direction is arbitrary and a model learns either. |
| Respiratory lag | **None.** The chest strap and the neck move together. |
| Where the new terms live | **`PulseStage`**, beside the vessel terms. It already owns the trace, the normalisation and the arterial delay the skin timing needs; a separate stage would take the same inputs. Its docstring becomes "physiology → pixel modulation". |
| Visibility guarantee | **Extended to the skin**: the retry loop also requires the background mask's green cardiac SNR ≥ 10, so a kept sample has a recoverable skin PPG as well as recoverable vessels. |
| Testing | **No new tests.** One fixture gains the stage's new argument mechanically. Verification is a region-mean plot and the validator (section 8). |

## 3. Skin PPG term

**Drive.** `ppg_n = normalise(trace[:, 4])`, the `[−0.5, 0.5]` percentile
map already used for the vessel drives. The trace's respiratory wander and
amplitude modulation survive normalisation, so the skin term carries
respiration too.

**Timing.** The stored PPG's foot is at `R + r_to_ppg_foot_s` (finger). The
neck skin fills just after the carotid: its foot is at

```text
r_to_skin_foot_s = pep_s + heart_to_neck_artery_m / pwv_art + skin_transit_s
```

where the middle term is the artery's entry delay, the value `delay_art`
takes at the vessel's caudal end, and `skin_transit_s` is a new
`TraceConfig` field, `Range(0.02, 0.04)`: the arteriole and capillary
transit from the carotid to the neck skin. The skin drive at frame time `t`
is `ppg_n(t + skin_lead_s)` with

```text
skin_lead_s = r_to_ppg_foot_s − r_to_skin_foot_s
```

With the existing ranges (`pep_s` 0.08–0.12, entry delay ≈ 0.03–0.05 at
`heart_to_neck_artery_m = 0.20` m and PWV 4–7 m/s) the neck skin foot lands
0.13–0.21 s after R, and leads the finger by 0.01–0.16 s. Both orderings hold
by construction: the skin lags the carotid entry by `skin_transit_s > 0`, and
the entry delay plus `skin_transit_s` maxes below the smallest `brachial + radial +
radial_to_finger` sum (0.09 < 0.10 s), so the finger is always distal.

**Sources.** Pulse arrival at the ear and forehead is 50–100 ms earlier than
at the finger, and forehead/ear PAT sits near 0.15–0.20 s at rest (Allen,
*Physiol Meas* 28:R1–R39, 2007, review; textbook values, not fetched). The
neck skin is treated like the forehead: a head-and-neck bed supplied by the
carotid. `skin_transit_s` is therefore a modelling choice within that band,
not a measured quantity; it is listed for Neckflix calibration in section 10.

**Amplitude.** `skin_amplitude_levels = amplitude_levels × skin_ratio`, a
`PulseParams` property beside `vein_amplitude_levels`. With the presets'
artery ranges this gives about 0.7–3.6 levels on `lesson` and 0.2–1.5 on
`benchmark`, against real green-channel rPPG of roughly 0.1–0.5 % of the DC
level (under one level at typical skin brightness). Like the vessel
amplitudes, `lesson` is deliberately larger than life.

**Pixel model.** The green modulation becomes

```text
G_mod = −( A_art · w_art · p_art  +  A_vein · w_vein · p_vein  +  A_skin · p_skin )
```

with `p_skin` a scalar per frame (spatially uniform; every pixel of the
cylinder is skin, the vessels included). `rgb_mod`, `ir_mod` are unchanged
in form: they scale `G_mod` by `channel_gain` and `ir_gain`. `depth_lift_mm`
does not include the skin term.

## 4. Respiration terms

`x(t) = 2·interp(t, trace[:, 0], trace[:, 5]) − 1`, in `[−1, 1]`, `+1` at
end-inspiration, evaluated at the frame time with no lag.

**Brightness.** `resp_gain(t) = 1 + resp_gain_frac · x(t)`, a scalar applied
multiplicatively to the whole rendered scene (RGB and IR, base plus pulse
terms) *before* the illumination stage: it is the scene tilting, not the
light changing. `PulseConfig.resp_gain_frac`, `Range(0.003, 0.010)`, peak
fractional change of 0.3–1 %, on the scale of the `benchmark` preset's
drift (0–2 % sd) and flicker (≤ 1 %).

**Depth.** The whole frame moves nearer by `resp_lift_mm · x(t)`. It is added
into `depth_lift_mm` as a scalar, so the renderer's depth line is unchanged.
`PulseConfig.resp_lift_mm`, `Range(0.5, 1.0)`: plain against the `lesson`
preset's 0.4 mm depth noise, and recoverable from the frame mean against the
default 1.6 mm.

Neither term has a literature anchor: both are stand-ins for the small
posture change of quiet breathing, sized to be learnable rather than
measured. They are listed for calibration in section 10.

## 5. Components

| Module | Change |
| --- | --- |
| `config.py` | `PulseConfig` gains `skin_ratio = Range(0.1, 0.3)`, `resp_gain_frac = Range(0.003, 0.010)`, `resp_lift_mm = Range(0.5, 1.0)`. `PulseParams` gains `skin_ratio = 0.2`, `resp_gain_frac = 0.005`, `resp_lift_mm = 0.7` and the property `skin_amplitude_levels`. `TraceConfig` gains `skin_transit_s = Range(0.02, 0.04)`; `TraceParams` gains `skin_transit_s = 0.03`. `sample()` is unchanged: it draws every field by name. |
| `render/pulse.py` | `PulseStage.__init__(pulse, gp, geometry, pixel_scale_mm, trace, timing: TraceParams)`; the `site_delay_s` keyword is removed (it is read from `timing.abp_site_delay_s`). The stage computes `ppg_n`, `rr` (the raw column), `entry_delay_art = gp.heart_to_neck_artery_m / pwv_art`, `r_to_skin_foot_s`, `skin_lead_s`. New methods: `skin_drive(time_s) -> float`, `resp_excursion(time_s) -> float` (the `x` of section 4), `resp_gain(time_s) -> float`. `_green_mod` adds the skin term; `depth_lift_mm` adds the respiratory offset. |
| `render/renderer.py` | Passes `params.trace` as `timing`. `frame` and `ir_frame` multiply `(base + pulse_mod)` by `self.pulse.resp_gain(t)` before the illumination stage. `depth_frame` is unchanged. |
| `generate.py` | `_render` also averages green over the background mask (`r.ids == 0`) and returns `(renderer, streams, artery_snr, vein_snr, skin_snr)`. The gate requires all three ≥ `MIN_VISIBLE_SNR`; the give-up message names all three. `visibility` gains `skin_snr`; `derived` gains `r_to_skin_foot_s` and `skin_lead_s` from the stage. |
| `inspect.py` | `vessel_snr` returns `(artery, vein, skin)` per channel, the skin being the background mask's mean. `inspect_root` prints the green skin SNR on each line. `vessel_power_ratio` is unchanged: artery and vein power relative to background now measures how much the vessels stand out *above* the skin PPG, and its values drop accordingly. |
| `zarr_store.py`, `folder_store.py`, `traces.py`, `presets.py`, `calibrate.py` | Unchanged. The presets inherit the new ranges from the defaults; `config_from_priors` builds the config blocks by keyword, so the new fields take their defaults. |
| `tests/test_render_pulse.py` | Mechanical: `_stage` passes its `TraceParams` as the stage's sixth argument. Not run. `test_inspect.py`'s artery-above-background ratios stay true: `((1 + r) / r)² ≥ 19` at `skin_ratio ≤ 0.3` on the artery pixels, well over its 1.5 threshold. `test_generate.py`'s visibility tests keep their `attempts == 2` shape with three SNR calls per attempt (the fake fails only the first two calls). Neither is edited. |
| `README.md` | A "What the frames carry" table under Traces (section 1's table), the skin timing sentence, the respiratory terms, and a note under Inspect that power ratios are now relative to the pulsing skin. |

## 6. Store and metadata delta

No change to the zarr layout, the arrays, the units or the root attrs. The
`synthetic_neck` metadata dict gains:

```text
derived.r_to_skin_foot_s   float
derived.skin_lead_s        float
visibility.skin_snr        float
```

## 7. Error handling

Unchanged in kind. A draw whose skin PPG is not visible is retried like a
draw whose vessels are not, from the same sub-stream sequence, and the
give-up `RuntimeError` reports the last three SNRs. Skin SNR is measured on
the mean of the largest region in the frame, so it is expected to pass on
every preset; if a preset starts failing the gate, `skin_ratio` is the knob.

## 8. Verification

No new tests. The one mechanical fixture edit is listed in the summary.

1. **Region plot.** A throwaway script in the session scratchpad generates
   two `lesson` samples and one `benchmark` sample with IR and depth as
   zarr stores, then plots over a 5 s window: the green mean of the artery,
   vein and background masks against the stored ABP, CVP and PPG (each
   normalised), and the depth frame mean against `rr`. Below them, the
   background green mean's spectrum. The check: the skin trace is a delayed
   PPG that leads the finger PPG and lags the carotid; the artery and vein
   traces now carry the skin pulse on top of their own; the depth mean and
   the frame brightness rise and fall with `rr`; the skin spectrum shows a
   peak at the heart rate and one at the breathing rate.
2. **Validator.** From the remote-physiology root:

   ```bash
   uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
       --preset lesson --n 4 --set video.frame_size=64 \
       --set streams.depth_ir_probability=0.5 --out <scratch>/synthetic_zarr
   uv run python tools/validate_cache.py <scratch>/synthetic_zarr
   ```

   must print `4/4 stores pass`, and each store's `synthetic_neck.visibility`
   must carry `skin_snr ≥ 10`.
3. **Inspect.** `synthetic-neck inspect` over a folder-mode `lesson` sample
   prints a skin SNR on each line and still produces `fft_maps.png`.

## 9. Changes in remote-physiology

- `dataset/data_loader/SYNTHETIC_NECK.md`: the "Frames" bullet describes
  the skin PPG and respiratory terms (the table in section 1); the
  "Guaranteed pulse" bullet adds the skin; the `vessel_ids` bullet's
  "nothing in the scene moves" becomes "nothing moves in-plane".
- Submodule pointer bumped to the commit that lands this change.

## 10. Out of scope

- Sub-pixel motion from breathing or pulse.
- A distinct neck-skin PPG morphology, or spatially varying perfusion.
- Calibrating `skin_ratio`, `skin_transit_s`, `resp_gain_frac` and
  `resp_lift_mm` from Neckflix.
- Rendering the ECG.
- Tuning the beat kernels (dicrotic hump size, early feet) noted after the
  traces pass.
