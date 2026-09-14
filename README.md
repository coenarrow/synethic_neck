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
The preset reads it from the repository's priors/ directory, so run the tool from a source checkout.

## Develop

    uv run pytest

Design: `docs/superpowers/specs/2026-09-14-synthetic-neck-generator-design.md`.
