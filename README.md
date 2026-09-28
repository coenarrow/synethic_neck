# synthetic-neck

Synthetic recordings of a neck with a pulsing carotid and jugular, with full ground truth, built to the geometry
and the streams of the Neckflix rig: a 650 px crop of a 90° camera at 500 to 1000 mm, delivering colour (uint8),
near-infrared (uint16) and depth (uint16, integer millimetres) at 30 Hz for 30 s, with the vessel masks, the
posture and five physiological traces on the same clock.

Every prior is drawn once per sample from `priors/base.yaml`, where each value is a fixed number, a uniform
`[lo, hi]` or a clipped normal `{MEAN, SD, MIN, MAX}`. The values come from the literature or from probes of the
Neckflix dataset, whose aggregated quantiles are in `priors/neckflix.json`; assumed values are marked as such.
The method is written up in two documents:

- [trace_generation.md](docs/trace_generation.md): the respiratory waveform, the R-wave walk, the ECG, the arterial
  pressure at the brachial or radial site, the central venous pressure and the finger PPG, all on one padded 1 kHz
  grid.
- [frame_rendering.md](docs/frame_rendering.md): the pinhole camera, the cylindrical neck with two straight vessels, the
  propagation of the carotid and CVP pulses along them, the static appearance (a Monk skin tone, Lambertian
  lighting, texture), the pressure-to-volume step that lifts the skin, the blood-to-colour step through the optics
  of Jacques, and the sensor.

The jugular pulses in every sample: its lift is set by the CVP and a measured venous compliance, and nothing is
redrawn or hidden.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

    uv sync

## Generate

    uv run synthetic-neck generate \
    --n 20 \ # how many samples
    --size 300 \ # resolution of samples (300x300)
    --jobs 2 \ # for concurrent generation
    --priors priors/base.yaml \ # params for generation
    --seed 2026 # random seed for reproducibility

Sample `i` uses seed `--seed + i` and is reproducible from it. 
`--size` delivers the frames area-averaged down from the native 650 px crop; without it they are delivered at 650 px.
To change a prior, copy `priors/base.yaml` and pass the copy with `--priors`.

Each sample is one zarr store `{i}.zarr` in the layout of remote-physiology's cache contract:

    {i}.zarr                  root attrs: participant, recording, posture, abp_site, monk_tone, seed,
                              synthetic_neck (every drawn value and the derived quantities)
    |-- vessel_ids            (H, W) uint8: 0 background, 1 artery, 2 vein
    |-- neck_mask             (H, W) uint8
    `-- 1/                    attrs: fps
        `-- rgb | ir | depth  video/data (C, T, H, W); timestamps_us/data (T,);
                              abp, cvp (mmHg), ecg (mV), ppg, rr (arb): <trace>/data (T,) on the frame clock

The dataset root gets `dataset.json` with the priors used, the seeds and any failures.

## Traces

| trace | units | what it is |
| --- | --- | --- |
| `rr` | arb | the respiratory waveform in [-1, 1], +1 at end-inspiration |
| `ecg` | mV | the ECGSYN beat on the R-wave walk, with respiratory modulation |
| `abp` | mmHg | arterial pressure at the drawn catheter site, brachial or radial (`abp_site`), with catheter noise |
| `cvp` | mmHg | central venous pressure with its a, c, x, v, y landmarks and catheter noise |
| `ppg` | arb | finger pleth, systole up, with respiratory amplitude modulation and wander |

The frames read the carotid pressure and the CVP without noise, delayed along each vessel, and a skin pulse that
carries the PPG's respiratory modulation; the stored traces are what a monitor would record.
