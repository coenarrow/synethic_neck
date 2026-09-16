# Physiological traces: ECG, PPG and respiration — design

Date: 2026-09-16
Status: approved in discussion

## 1. Purpose

Every model remote-physiology trains predicts ABP, CVP, ECG, PPG and
respiration together. The generator produces only ABP and CVP, so a synthetic
cache can train and validate two of the five targets. This change makes the
generator emit all five, from one cardiac timeline and one respiratory
waveform, with the delays between them matching resting-adult physiology, so
the synthetic dataset can stand in for a real one end to end before Neckflix
and later datasets are cached.

This pass is **traces only**. The frames change in exactly one place (section
5), to keep the carotid pixels correctly timed against the new traces.
Skin-wide PPG modulation of the frames is a separate, later change.

## 2. Decisions

| Question | Decision |
| --- | --- |
| Where the PPG and ECG go | Two new columns of the 1 kHz trace, written to the store as `ecg` and `ppg` trace groups beside `abp` and `cvp`. |
| Respiration | A sixth column `rr`: a chest-strap style excursion in `[0, 1]`, rising on inspiration, units `"arb"`. Neckflix has no respiration trace; only its frequency matters. |
| ECG model | The Gaussian-sum (closed-form) waveform of the **McSharry ECGSYN model** (McSharry, Clifford, Tarassenko, Smith, *IEEE Trans Biomed Eng* 50(3):289–294, 2003). Documented in the README and the module docstring. |
| Stored ABP site | Drawn per sample from `radial` and `brachial`, recorded as the root attr `abp_site`. Neckflix contains both. |
| Render drive for the carotid | The stored ABP shifted **earlier** by the site delay (approach A): the pixels stay a pure function of the stored ground truth. No unstored drive signals. |
| Timing sources | Literature ranges (section 4), not a re-derivation from MIMIC. |
| Testing | No new tests. Verification is a trace plot for eyeballing and the validator (section 8). |

## 3. Trace model

`generate_trace(p: TraceParams) -> (N, 6)` with columns
`Time, ABP, CVP, ECG, PPG, RR`, all at `sample_rate_hz` (1 kHz). One
`np.random.Generator` seeded from `p.seed`, as today.

**Respiration.** `x(t) = sin(2π f_resp t + φ)` with `f_resp` from
`resp_rate_bpm` and a drawn phase `resp_phase_rad`. The stored trace is
`rr = 0.5 + 0.5 x`. `x = +1` is end-inspiration.

**Cardiac timeline.** `beat_onsets` keeps its Gaussian RR jitter and gains
respiratory sinus arrhythmia: the interval that starts at R-wave `r_k` is
`RR_mean × (1 + hr_variability ε_k − rsa_fraction x(r_k))`, so heart rate
rises on inspiration. R-wave times remain the anchor for every other signal.

**ECG (mV).** Per beat `k`, the angle `θ = 2π (t − r_k) / RR_k` and
`z(θ) = Σ_i a_i b_i² exp(−(θ − θ_i)² / 2 b_i²)` over the five ECGSYN waves
with the published defaults `θ = (−70°, −15°, 0°, 15°, 100°)`,
`a = (1.2, −5, 30, −7.5, 0.75)`, `b = (0.25, 0.1, 0.1, 0.1, 0.4)`. This is the
exact integral of ECGSYN's `dz/dt` around the limit cycle with the baseline
relaxation term dropped; wave positions therefore scale with each beat's RR,
as in ECGSYN. The beat sum is scaled so the R-peak equals `r_amplitude_mv`.
Respiration enters as R-amplitude modulation `× (1 + ecg_r_modulation x)`
and baseline wander `+ ecg_wander_mv x`; white noise `ecg_noise_mv`.

**ABP (mmHg).** The existing central beat shape (`_abp_beat`), whose upstroke
sits at `r_k + pep_s`. `pep_s` is the renamed `abp_upstroke_delay_s`. The
stored trace is that waveform delayed by the site transit:
`brachial_transit_s` for brachial, `brachial_transit_s + radial_transit_s`
for radial. The central waveform is scaled to the drawn systolic/diastolic
pair (brachial values); for radial the pulse pressure is multiplied by
`radial_amplification` about the diastolic level. Respiratory swing:
`− resp_abp_swing_mmhg x` (falls on inspiration; today's sign was the
reverse, which is the mechanically ventilated pattern). Noise as today.

**CVP (mmHg).** Unchanged: a, c, x, v, y waves anchored on `r_k`, swing
`− resp_cvp_swing_mmhg x`, noise, clip to `[2, 20]`.

**PPG (arb).** A new peripheral beat shape `_ppg_beat(phase)`: a broad
systolic hump peaking about 0.18 s after the foot, a dicrotic hump near
0.40 s and a slow exponential run-off. The foot of beat `k` is at
`r_k + pep_s + finger_transit_s`. The beat sum is mapped to `[0, 1]` by its
min and max, then amplitude-modulated about 0.5 by `(1 + ppg_am_frac x)`,
baseline-shifted by `ppg_wander_frac x`, plus white noise `ppg_noise`.

**Derived delays** (properties on `TraceParams`, also written to metadata
under `derived`):

| Property | Value |
| --- | --- |
| `abp_site_delay_s` | `brachial_transit_s` or `brachial_transit_s + radial_transit_s` |
| `r_to_abp_foot_s` | `pep_s + abp_site_delay_s` |
| `r_to_ppg_foot_s` | `pep_s + finger_transit_s` |

## 4. Timing anchors and their sources

Defaults are `Range`s in `TraceConfig`. Every preset uses them; `neckflix`
does not calibrate them this pass.

| Field | Default range | What it is | Source |
| --- | --- | --- | --- |
| `pep_s` | 0.08–0.12 s | R-wave to aortic valve opening (pre-ejection period) | Weissler et al. 1968 regressions give ≈100 ms at resting heart rates; textbook value |
| `brachial_transit_s` | 0.05–0.09 s | aortic root to brachial artery | ≈0.5 m path at aortic–brachial PWV of 6–10 m/s; textbook value |
| `radial_transit_s` | 0.02–0.04 s | brachial to radial | brachial–radial PTT 26.8 ± 6.4 ms (Regional variation in PTT in the upper limb, PMC12095889) |
| `radial_amplification` | 1.05–1.15 | brachial→radial pulse-pressure amplification | 12 ± 11 % (Verbeke et al. 2005, PMID 15911747) |
| `finger_transit_s` | 0.12–0.20 s | aortic root to finger PPG foot | with `pep_s` gives PAT-to-foot of 0.20–0.32 s, the band reported for resting adults in MIMIC-based PAT studies (e.g. Liang et al. 2019, *J Clin Med* 8(3):337); studies quoting PAT to the PPG *peak* report ≈0.4 s, consistent with a ≈0.18 s crest time |
| `rsa_fraction` | 0.02–0.08 | peak RR shortening on inspiration | RSA of a few percent of RR in resting adults, larger when young (Circulation 94:842, 1996) |
| `r_amplitude_mv` | 0.8–1.5 mV | R-peak height | lead-II textbook value |
| `ecg_r_modulation` | 0.03–0.10 | respiratory R-amplitude modulation | ECG-derived-respiration literature: a few to ten percent |
| `ecg_wander_mv` | 0.02–0.08 mV | respiratory baseline wander | same |
| `ecg_noise_mv` | 0.005–0.02 mV | white noise | monitor-grade |
| `ppg_am_frac` | 0.05–0.15 | respiratory amplitude modulation | PPG-derived-respiration literature |
| `ppg_wander_frac` | 0.05–0.15 | respiratory baseline wander | same |
| `ppg_noise` | 0.005–0.02 arb | white noise | small |
| `resp_phase_rad` | 0–2π | breathing phase at t = 0 | uniform |
| `abp_site` | `Choice(("radial", "brachial"))` | catheter site | Neckflix has both |

ECGSYN wave parameters: PhysioNet ECGSYN 1.0.0
(<https://physionet.org/content/ecgsyn/1.0.0/>) and McSharry et al. 2003.

Resulting relationships at the defaults: CVP a-wave 80 ms before R; carotid
upstroke ≈ R + 0.10 s; brachial foot R + 0.13–0.21 s; radial foot
R + 0.15–0.25 s; finger PPG foot R + 0.20–0.32 s; PPG peak ≈ foot + 0.18 s;
T-wave peak at 100/360 of the RR interval (≈ R + 0.23 s at 72 bpm).

## 5. Components

| Module | Change |
| --- | --- |
| `config.py` | `TraceConfig` gains the fields of section 4; `abp_upstroke_delay_s` is renamed `pep_s`. `TraceParams` gains the matching scalars, `abp_site: str`, and the three derived properties. `sample()` is unchanged: it draws every `TraceConfig` field by name. |
| `traces.py` | `TRACE_COLUMNS` becomes six names. `beat_onsets` takes the respiratory phase. `ECG_WAVES` module constant with the McSharry citation; `_ecg_beat`, `_ppg_beat` beside `_abp_beat`, `_cvp_beat`. CSV writer/reader carry six columns (Time 4 dp, ABP and CVP 3 dp, ECG, PPG and RR 4 dp). |
| `render/pulse.py` | `PulseStage` takes `site_delay_s` and interpolates the artery at `time_s − delay_art + site_delay_s`. Nothing else in `render/` changes. |
| `render/renderer.py` | Passes `params.trace.abp_site_delay_s` to `PulseStage`. |
| `zarr_store.py` | `TRACE_COLUMNS = {abp: 1, cvp: 2, ecg: 3, ppg: 4, rr: 5}`, `TRACE_UNITS` adds `ecg: "mV"`, `ppg: "arb"`, `rr: "arb"`; root attr `abp_site` from `params["trace"]["abp_site"]`. |
| `folder_store.py` | Unchanged; `trace.csv` gains the columns through `traces.py`. |
| `generate.py` | `derived` gains `abp_site`, `abp_site_delay_s`, `r_to_abp_foot_s`, `r_to_ppg_foot_s`. |
| `presets.py`, `calibrate.py` | Unchanged. `config_from_priors` builds `TraceConfig` by keyword, so the new fields take their defaults. |
| `README.md` | Trace description: six columns, the model per signal, the McSharry citation, `abp_site`. |

## 6. Store layout delta

Under every modality group, beside `abp` and `cvp`:

```text
ecg/data   (T,) float64   attrs {units: "mV"}
ppg/data   (T,) float64   attrs {units: "arb"}
rr/data    (T,) float64   attrs {units: "arb"}
```

Root attrs gain `abp_site: "radial" | "brachial"`. Everything else in the
zarr design (2026-09-15) stands.

## 7. Error handling

Unchanged. `abp_site` is a `Choice` over the two admitted strings, so no new
refusal exists. The SNR gate measures the vessel pixels' own cardiac power
and is not affected.

## 8. Verification

No new tests. Existing trace tests that merely name the old field or unpack
three columns are edited mechanically (rename; `[:3]`; the shape and header
literals) and not run; they are listed in the summary.

1. **Trace plot.** A throwaway script in the session scratchpad generates
   three samples' traces (radial and brachial, different postures) and plots
   a 5 s window of all six signals, R-wave times marked, to a PNG. The check:
   CVP a-wave before R, carotid-timed ABP central upstroke after R, brachial
   then radial foot order and gaps, PPG foot after the ABP foot, T-wave
   before the ABP foot, heart rate visibly faster on inspiration against
   `rr`, ABP and CVP dipping together on inspiration.
2. **Validator.** From the remote-physiology root:

   ```bash
   uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
       --preset lesson --n 4 --set video.frame_size=64 \
       --set streams.depth_ir_probability=0.5 --out <scratch>/synthetic_zarr
   uv run python tools/validate_cache.py <scratch>/synthetic_zarr
   ```

   must print `4/4 stores pass`; every modality now carries five traces
   with `units`.
3. **Folder mode.** The same command without `--zarr` still writes the five
   files, and `trace.csv` has the six-column header.

## 9. Changes in remote-physiology

- `dataset/data_loader/SYNTHETIC_NECK.md`: the five trace groups, their units
  and models, and `abp_site` under root attributes.
- `configs/datasets/synthetic_neck.yaml`: `abp_site radial | brachial` in
  the filterable-attrs header.
- Submodule pointer bumped to the commit that lands this change.

## 10. Out of scope

- Skin-wide PPG modulation of the frames (next pass).
- Calibrating the new ranges from Neckflix.
- Mechanical-ventilation swing polarity, non-sinusoidal breathing, breathing
  rate drift, arrhythmia.
- Distinct neck-skin and finger PPG morphologies.
