# synthetic-neck

Generates synthetic videos of a neck with propagating carotid (arterial) and
jugular (venous) pulsation, with full ground truth. Three presets:

| Preset | Use | Pulse amplitude |
|---|---|---|
| `lesson` | teaching: obvious signal, flat lighting | 7–12 levels |
| `benchmark` | signal-processing benchmarks with controlled degradation | 2–5 levels |
| `neckflix` | ranges calibrated from the Neckflix ICU dataset | 0.5–2 levels |

Every sample is guaranteed a testable pulse. While rendering, the generator measures the green channel's vessel-averaged cardiac signal-to-noise ratio for the artery and for the vein. It tries up to 20 draws of the sample's settings until both reach at least 10. The infrared stream is not guaranteed to show a pulse. Each sample's `metadata.json` records the outcome under `visibility`.

## Install

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on PATH.

    uv sync --all-extras

## Generate

    uv run synthetic-neck generate --preset lesson --n 5 --out data/synthetic_necks
    uv run synthetic-neck generate --preset neckflix --n 20 --jobs 4 --set pulse.amplitude_levels=1,1.5

`--set block.field=value` overrides any config field. Ranges are `lo,hi` or a
single fixed value; choices are `a|b|c`. Unknown names fail before rendering.

Each sample directory contains `trace.csv` (Time, ABP, CVP, ECG, PPG, RR at 1 kHz; see Traces),
`gray_video.mkv`, `rgbid_video.mkv` (streams `rgb`, and when present `ir`
and `depth` in 0.02 mm units), `vessel_ids.npy` (0 background, 1 artery,
2 vein) and `metadata.json` (effective config, drawn parameters, derived
quantities). The dataset root gets `dataset.json`.

### Zarr stores for remote-physiology

    uv run synthetic-neck generate --zarr --preset neckflix --n 20 --out data/synthetic_zarr

`--zarr` writes one `{i}.zarr` store per sample instead of the folder, in the layout of remote-physiology's cache
contract (`docs/cache-contract.md` there). Perspective `1` holds `rgb`, plus `ir` (uint8) and `depth` (float32 mm)
when the sample has them. Each modality carries per-frame timestamps and the five traces (`abp`, `cvp` in mmHg,
`ecg` in mV, `ppg` and `rr` in arb) interpolated to the frames. Root attrs hold `participant` (the sample
index), `posture`, `abp_site`, `monk_tone`, `preset`, `seed` and the full
metadata under `synthetic_neck`; `vessel_ids` is a root array. ffmpeg is not needed, and `trace.posture_deg`
must stay within 0, 45 and 90. Design: `docs/superpowers/specs/2026-09-15-zarr-output-design.md`.

## Traces

Every sample carries five ground-truth traces at 1 kHz, generated from one
cardiac timeline (R-wave times with respiratory sinus arrhythmia) and one
respiratory sinusoid, so the delays between them are fixed by construction:

| Column | Units | Model |
| --- | --- | --- |
| `ABP` | mmHg | central pulse (systolic peak, dicrotic wave, run-off) delayed to a drawn catheter site, `radial` or `brachial`, recorded as `trace.abp_site`; radial pulse pressure is amplified |
| `CVP` | mmHg | a, c, x, v, y waves anchored on the R-wave |
| `ECG` | mV | the McSharry ECGSYN model (McSharry, Clifford, Tarassenko, Smith, *IEEE Trans Biomed Eng* 50(3):289–294, 2003) as its closed-form Gaussian sum, scaled to the drawn R amplitude |
| `PPG` | arb | finger pulse: broad systolic hump, dicrotic hump, run-off; foot at R + `pep_s` + `brachial_transit_s` + `radial_transit_s` + `radial_to_finger_s` |
| `RR` | arb | chest excursion in [0, 1], rising on inspiration |

Respiration modulates everything: ABP and CVP fall on inspiration, ECG
gets baseline wander and R-amplitude modulation, PPG gets baseline wander and
amplitude modulation, and beat-to-beat intervals shorten on inspiration. The timing
fields and their literature sources are tabulated in
`docs/superpowers/specs/2026-09-16-physiological-traces-design.md`. The
carotid pixels are rendered from ABP shifted back to central timing.

## Inspect

    uv run synthetic-neck inspect --root data/synthetic_necks

Writes `fft_maps.png` per sample and prints artery/vein power at the heart
rate relative to background.

## Calibrate (needs the Neckflix drive)

    uv run synthetic-neck calibrate --root "/Volumes/Blue 4TB/CVP/Dataset/Neckflix" --out priors/neckflix.json

Writes aggregated quantiles and histograms only (no IDs, no pixels). The
checked-in `priors/neckflix.json` is what `--preset neckflix` reads.
The preset reads it from the repository's priors/ directory, so run the tool from a source checkout.

## Develop

    uv run pytest

Design: `docs/superpowers/specs/2026-09-14-synthetic-neck-generator-design.md`.
