# `generate --zarr` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `synthetic-neck generate --zarr` writes one zarr store per sample
that passes remote-physiology's `tools/validate_cache.py`, in place of the
folder layout, and leaves folder-mode output unchanged.

**Architecture:** `_render` stops writing files and drives a sink.
`FolderSink` (today's MKV/CSV code, moved) and `ZarrSink` (new) implement
the same four calls. `ZarrSink` streams chunk-sized frame blocks into
`{i}.zarr.partial` and renames it to `{i}.zarr` only after the SNR check
passes. remote-physiology then gains a dataset config and a cache spec for
these stores, plus the bumped submodule pointer.

**Tech Stack:** Python 3.13 and uv; numpy; zarr 3 (`zarr.codecs.BloscCodec`,
`zarr.codecs.numcodecs.Delta`); ffmpeg/ffprobe (folder mode only).

**Spec:** `docs/superpowers/specs/2026-09-15-zarr-output-design.md`, in this
repository.

## Global Constraints

- **No tests.** No new test files and no pytest runs. The only checks are
  running `synthetic-neck`, inspecting its output (`ls`, `shasum`,
  `ffmpeg -f framemd5`, `diff`), and remote-physiology's
  `uv run python tools/validate_cache.py <dir>`.
- **Working directories.** Paths are given relative to the remote-physiology
  root (`/Users/20759193/repos/remote-physiology`), which carries this
  repository at `tools/synthetic_datasets/synthetic_neck`. Run the generator
  from there with `uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck ...`.
- **`$SCRATCH`.** A scratch directory outside both repositories; in a Claude
  Code session, use the session scratchpad. Generated output only ever goes
  there.
- **Dependencies.** The only new one is `zarr>=3.3,<4`, added with `uv add`
  (never pip).
- **Folder mode is unchanged.** Without `--zarr`, every sample's decoded
  frames, `trace.csv`, `vessel_ids.npy` and `metadata.json` must be identical
  to before the change. MKV bytes are not compared, because Matroska writes a
  random segment UID.
- **Contract facts.** `participant` is a string; perspective `"1"`; modality
  keys `rgb`/`ir`/`depth`; trace keys `abp`/`cvp` with `units` `"mmHg"`; every
  array at `<name>/data`; `fps` on the perspective.
- **Axis moves.** numpy `stack`/`transpose`: this repository is numpy-only,
  and remote-physiology's einops rule does not apply here.
- **Branches.** This repository: `feat/zarr-output`, which already carries
  the spec commit. remote-physiology: `feat/synthetic-neck-zarr`. Ask before
  pushing either.
- **Commits.** This repository uses plain sentence-case messages;
  remote-physiology uses conventional commits. Every commit ends with
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File structure

| File | Responsibility |
| --- | --- |
| `src/synthetic_neck/folder_store.py` (new) | `FolderSink`: the per-sample folder layout (`trace.csv`, MKVs, `vessel_ids.npy`, `metadata.json`) |
| `src/synthetic_neck/zarr_store.py` (new) | `ZarrSink` and `check_config`: the cache-contract store |
| `src/synthetic_neck/generate.py` | Seeding, the SNR retry loop, metadata, `dataset.json`; picks the sink |
| `src/synthetic_neck/cli.py` | The `--zarr` flag |
| `README.md` | Documents `--zarr` |
| `pyproject.toml`, `uv.lock` | The `zarr` dependency |
| remote-physiology `configs/datasets/synthetic_neck.yaml` (new) | Dataset config: `CACHED_PATH`, `FILTERS`, the filterable attrs |
| remote-physiology `dataset/data_loader/SYNTHETIC_NECK.md` (new) | Cache spec for these stores |

---

### Task 1: Move folder output behind a sink

A pure refactor: folder output must come out identical.

**Files:**

- Create: `src/synthetic_neck/folder_store.py`
- Modify: `src/synthetic_neck/generate.py` (whole file shown below)

**Interfaces:**

- Consumes: `MkvWriter`, `mux`, `require_ffmpeg` (`video.py`);
  `depth_to_uint16`, `to_gray` (`render/camera.py`);
  `write_trace_csv`, `read_trace_csv` (`traces.py`).
- Produces the sink protocol that Task 2 implements a second time:
  - `begin(trace: np.ndarray, streams: list[str], n_frames: int, frame_size: int, fps: float) -> np.ndarray`
  - `frame(rgb: np.ndarray, ir: np.ndarray | None, depth_mm: np.ndarray | None) -> None`
  - `commit(vessel_ids: np.ndarray, meta: dict) -> None`
  - `discard() -> None`

  Also `FolderSink(out_dir: Path)`, and in `generate.py`:
  `_render(params, sink)`, `generate_sample(config, seed, out_dir, preset="custom")`,
  `_one(args)` with `args = (config, index, seed, out_dir, preset)`.

- [ ] **Step 1: Record a folder-mode baseline with the unchanged code**

Run from the remote-physiology root, before editing anything:

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate \
    --preset lesson --n 4 --set video.frame_size=64 --set streams.depth_ir_probability=0.5 \
    --out "$SCRATCH/folder_before"
```

Expected: `wrote 4/4 samples to .../folder_before`.

- [ ] **Step 2: Create `src/synthetic_neck/folder_store.py`**

```python
"""Per-sample folder output: trace.csv, gray_video.mkv, rgbid_video.mkv, vessel_ids.npy, metadata.json."""
from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path

import numpy as np

from .render.camera import depth_to_uint16, to_gray
from .traces import read_trace_csv, write_trace_csv
from .video import MkvWriter, mux, require_ffmpeg


class FolderSink:
    """Writes one sample into `out_dir`: trace.csv, gray_video.mkv, rgbid_video.mkv (stream rgb, plus ir and
    depth in 0.02 mm units when the sample has them), vessel_ids.npy and metadata.json."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self._writers: list[MkvWriter] = []
        self._streams: list[str] = []

    def _temp(self, stream: str) -> Path:
        return self.out_dir / f"_{stream}.mkv"

    def begin(self, trace: np.ndarray, streams: list[str], n_frames: int, frame_size: int,
              fps: float) -> np.ndarray:
        require_ffmpeg()
        for w in self._writers:     # an earlier attempt that failed the SNR check; its files are overwritten
            w.close()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # The video is rendered from the CSV on disk, so the pixels match the ground truth as written.
        write_trace_csv(trace, self.out_dir / "trace.csv")
        size = (frame_size, frame_size)
        self._streams = list(streams)
        self._writers = [MkvWriter(self._temp("rgb"), size, "rgb", fps),
                         MkvWriter(self.out_dir / "gray_video.mkv", size, "gray", fps)]
        if "ir" in streams:
            self._writers += [MkvWriter(self._temp("ir"), size, "gray", fps),
                              MkvWriter(self._temp("depth"), size, "gray16", fps)]
        return read_trace_csv(self.out_dir / "trace.csv")

    def frame(self, rgb: np.ndarray, ir: np.ndarray | None, depth_mm: np.ndarray | None) -> None:
        self._writers[0].write(rgb)
        self._writers[1].write(to_gray(rgb))
        if ir is not None:
            self._writers[2].write(ir)
            self._writers[3].write(depth_to_uint16(depth_mm))

    def commit(self, vessel_ids: np.ndarray, meta: dict) -> None:
        writers, self._writers = self._writers, []
        for w in writers:
            w.close()
        parts = [self._temp(s) for s in self._streams]
        mux(parts, self.out_dir / "rgbid_video.mkv", titles=self._streams)
        for tmp in parts:
            tmp.unlink()
        np.save(self.out_dir / "vessel_ids.npy", vessel_ids)
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def discard(self) -> None:
        writers, self._writers = self._writers, []
        for w in writers:
            with contextlib.suppress(Exception):
                w.close()
        shutil.rmtree(self.out_dir, ignore_errors=True)
```

- [ ] **Step 3: Replace `src/synthetic_neck/generate.py` with the sink-driven version**

```python
"""Generate one sample or a whole dataset."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import GeneratorConfig, config_to_dict, params_to_dict, sample, validate
from .folder_store import FolderSink
from .inspect import cardiac_snr
from .render.camera import DEPTH_UNITS_MM
from .render.renderer import Renderer
from .traces import generate_trace
from .video import require_ffmpeg

MIN_VISIBLE_SNR = 10.0
MAX_VISIBILITY_ATTEMPTS = 20


class SampleFailed(Exception):
    def __init__(self, index: int, seed: int, cause: Exception):
        super().__init__(f"sample {index} (seed {seed}) failed: {cause!r}")
        self.index, self.seed, self.cause = index, seed, cause


def _attempt_rng(seed: int, attempt: int) -> np.random.Generator:
    """Attempt 0 keeps the original stream for `seed`; later attempts use independent sub-streams."""
    return np.random.default_rng(seed if attempt == 0 else np.random.SeedSequence([seed, attempt]))


def _render(params, sink) -> tuple[Renderer, list[str], float, float]:
    """Render every stream in one pass into `sink`; return (renderer, streams, green artery SNR, green vein SNR)."""
    has_di = params.streams.has_depth_ir
    streams = ["rgb", "ir", "depth"] if has_di else ["rgb"]
    fps = params.video.fps
    # The sink decides which trace the video is rendered from (FolderSink: the CSV read back off disk).
    trace = sink.begin(generate_trace(params.trace), streams, params.video.n_frames, params.video.frame_size, fps)
    r = Renderer(params, trace)
    art_mask, vein_mask = r.ids == 1, r.ids == 2
    g_art, g_vein = np.empty(r.n_frames), np.empty(r.n_frames)
    for i in range(r.n_frames):
        rgb = r.frame(i)
        g = rgb[..., 1].astype(np.float64)
        g_art[i], g_vein[i] = g[art_mask].mean(), g[vein_mask].mean()
        # Arguments evaluate left to right, keeping the sensor noise draws in rgb -> ir -> depth order.
        sink.frame(rgb, r.ir_frame(i) if has_di else None, r.depth_frame(i) if has_di else None)
    hr_hz = params.trace.heart_rate_bpm / 60.0
    return r, streams, cardiac_snr(g_art, fps, hr_hz), cardiac_snr(g_vein, fps, hr_hz)


def generate_sample(config: GeneratorConfig, seed: int, out_dir: Path, preset: str = "custom") -> dict:
    """Render one sample into `out_dir` and return its metadata dict. Guarantees a testable green-channel
    pulse: both vessels are checked and, if either is under MIN_VISIBLE_SNR, the sample is redrawn from an
    independent sub-stream of `seed`, up to MAX_VISIBILITY_ATTEMPTS times. On failure nothing is left behind."""
    validate(config)
    sink = FolderSink(out_dir)
    try:
        last = (0.0, 0.0)
        for attempt in range(MAX_VISIBILITY_ATTEMPTS):
            params = sample(config, _attempt_rng(seed, attempt))
            r, streams, artery_snr, vein_snr = _render(params, sink)
            last = (artery_snr, vein_snr)
            if artery_snr >= MIN_VISIBLE_SNR and vein_snr >= MIN_VISIBLE_SNR:
                break
        else:
            raise RuntimeError(f"seed {seed}: no draw reached green vessel SNR {MIN_VISIBLE_SNR} in "
                               f"{MAX_VISIBILITY_ATTEMPTS} attempts (last artery {last[0]:.1f}, vein {last[1]:.1f})")

        g, cam, pulse = r.geometry, params.camera, r.pulse
        meta = {
            "seed": seed,
            "preset": preset,
            "version": __version__,
            "frame_size": params.video.frame_size,
            "fps": params.video.fps,
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
            "visibility": {"channel": "G", "min_snr": MIN_VISIBLE_SNR, "attempts": attempt + 1,
                           "artery_snr": artery_snr, "vein_snr": vein_snr},
        }
        sink.commit(r.ids, meta)
    except BaseException:
        sink.discard()
        raise
    return meta


def _one(args) -> tuple[int, Exception | None]:
    config, index, seed, out_dir, preset = args
    try:
        generate_sample(config, seed, out_dir, preset)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller; the sink already removed its output
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
        "failed": [{"index": i, "seed": base_seed + i} for i, e in results if e is not None],
    }
    (out_root / "dataset.json").write_text(json.dumps(index, indent=2))
    for i, e in results:
        if e is not None:
            print(f"sample {i} (seed {base_seed + i}) failed: {e!r}", file=sys.stderr)
    return results
```

- [ ] **Step 4: Regenerate with the refactored code**

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate \
    --preset lesson --n 4 --set video.frame_size=64 --set streams.depth_ir_probability=0.5 \
    --out "$SCRATCH/folder_after"
```

Expected: `wrote 4/4 samples to .../folder_after`.

- [ ] **Step 5: Compare the two outputs**

```bash
for d in before after; do
  ( cd "$SCRATCH/folder_$d" && for s in */; do
      ls "$s"
      for f in trace.csv vessel_ids.npy metadata.json; do echo "$s$f $(shasum -a 256 < "$s$f")"; done
      for f in gray_video.mkv rgbid_video.mkv; do
        echo "$s$f"; ffmpeg -v error -i "$s$f" -map 0 -f framemd5 - | grep -v '^#'
      done
    done ) > "$SCRATCH/folder_$d.txt"
done
diff "$SCRATCH/folder_before.txt" "$SCRATCH/folder_after.txt" && echo IDENTICAL
```

Expected: `IDENTICAL`. Each sample directory lists exactly `gray_video.mkv`,
`metadata.json`, `rgbid_video.mkv`, `trace.csv` and `vessel_ids.npy`, with no
leftover `_rgb.mkv`, `_ir.mkv` or `_depth.mkv`. Any difference is a refactor
bug; fix it before committing. The usual culprit is sensor noise drawn in a
different order.

- [ ] **Step 6: Commit (in this repository)**

```bash
git -C tools/synthetic_datasets/synthetic_neck add src/synthetic_neck/folder_store.py src/synthetic_neck/generate.py
git -C tools/synthetic_datasets/synthetic_neck commit -F - <<'EOF'
Move folder output behind a sink

_render now drives a sink (begin/frame/commit/discard) instead of writing
files itself; FolderSink holds the trace.csv, MKV, vessel_ids.npy and
metadata.json writing, unchanged, and is the only caller of require_ffmpeg
per sample. A failed sample is removed by generate_sample itself rather than
only by the dataset loop. Decoded frames and every other file are identical
to before for the same seeds.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: `ZarrSink` and the `--zarr` flag

**Files:**

- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `src/synthetic_neck/zarr_store.py`
- Modify: `src/synthetic_neck/generate.py` (imports, `generate_sample`,
  `_one`, `generate_dataset`)
- Modify: `src/synthetic_neck/cli.py` (the `generate` parser and its branch in `main`)
- Modify: `README.md` (after the Generate section's output paragraph)

**Interfaces:**

- Consumes: the Task 1 sink protocol; `ConfigError`, `GeneratorConfig`
  (`config.py`); `generate_trace`'s `(N, 3)` array with columns Time [s],
  ABP [mmHg], CVP [mmHg]; `meta["params"]["trace"]["posture_deg"]`,
  `meta["params"]["appearance"]["monk_tone"]`, `meta["preset"]` and
  `meta["seed"]` (`params_to_dict` nests each params dataclass by field
  name).
- Produces:
  - `ZarrSink(out_dir: Path)`, which writes `out_dir.with_name(out_dir.name + ".zarr")`
  - `check_config(config: GeneratorConfig) -> None`, which raises `ConfigError`
  - `generate_sample(config, seed, out_dir, preset="custom", *, zarr: bool = False) -> dict`
  - `generate_dataset(config, out_root, n, start, base_seed, preset, overrides, jobs=1, *, zarr: bool = False)`
  - CLI flag `--zarr`, and `dataset.json` key `"format"`

- [ ] **Step 1: Add the dependency**

From the remote-physiology root. `--project` avoids a `cd` that would leave
later steps running from the wrong directory.

```bash
uv add --project tools/synthetic_datasets/synthetic_neck "zarr>=3.3,<4"
```

Expected: `pyproject.toml` dependencies become `["numpy>=2.0", "zarr>=3.3,<4"]`
and `uv.lock` updates.

- [ ] **Step 2: Create `src/synthetic_neck/zarr_store.py`**

```python
"""Per-sample zarr output: one store satisfying remote-physiology's cache contract (docs/cache-contract.md there).

    {i}.zarr                  root attrs: participant, recording, posture, monk_tone, preset, seed, synthetic_neck
    |-- vessel_ids            (H, W) uint8, attrs: labels
    `-- 1/                    attrs: fps
        `-- rgb | ir | depth  video/data (C, T, H, W); timestamps_us/data (T,) int64;
                              abp/data, cvp/data (T,) float64 with units "mmHg"

Design: docs/superpowers/specs/2026-09-15-zarr-output-design.md.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

from .config import ConfigError, GeneratorConfig

PERSPECTIVE = "1"
CHUNK_FRAMES = 32
POSTURE_BY_ANGLE = {0.0: "supine", 45.0: "recumbent", 90.0: "sitting"}   # Neckflix's vocabulary
TRACE_COLUMNS = {"abp": 1, "cvp": 2}                                     # columns of generate_trace's array
TRACE_UNITS = {"abp": "mmHg", "cvp": "mmHg"}
VESSEL_LABELS = {"0": "background", "1": "artery", "2": "vein"}
MODALITY_FORMAT = {"rgb": (3, np.uint8), "ir": (1, np.uint8), "depth": (1, np.float32)}   # (channels, dtype)
# Settings of remote-physiology's cachers (dataset/cachers/*/writer.py), so every cache reads alike.
_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")


def check_config(config: GeneratorConfig) -> None:
    """Refuse, before anything renders, a config that could draw a value the store has no spelling for."""
    unknown = sorted({float(v) for v in config.trace.posture_deg.values} - set(POSTURE_BY_ANGLE))
    if unknown:
        raise ConfigError(f"--zarr writes posture as supine/recumbent/sitting, i.e. trace.posture_deg 0, 45 "
                          f"or 90; this config can draw {unknown}")


class ZarrSink:
    """Writes one sample as `{out_dir}.zarr`, where `out_dir` is the folder the sample would otherwise get.

    Frames stream into `{out_dir}.zarr.partial`, one chunk at a time; `commit` renames it into place, so a
    visible `*.zarr` store is always complete.
    """

    def __init__(self, out_dir: Path):
        out_dir = Path(out_dir)
        self.name = out_dir.name                           # the sample index: participant and recording
        self.path = out_dir.with_name(f"{self.name}.zarr")
        self.partial = out_dir.with_name(f"{self.name}.zarr.partial")
        self._root = None
        self._trace = None
        self._fps = 0.0
        self._n_frames = 0
        self._chunk = CHUNK_FRAMES
        self._videos = {}
        self._buffers = {}
        self._written = {}

    def begin(self, trace: np.ndarray, streams: list[str], n_frames: int, frame_size: int,
              fps: float) -> np.ndarray:
        shutil.rmtree(self.partial, ignore_errors=True)
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(str(self.partial), mode="w")
        self._trace, self._fps, self._n_frames = trace, float(fps), n_frames
        self._chunk = min(CHUNK_FRAMES, n_frames)
        perspective = self._root.create_group(PERSPECTIVE)
        perspective.attrs["fps"] = self._fps
        timestamps_us = np.round(np.arange(n_frames) * 1e6 / self._fps).astype(np.int64)
        self._videos, self._buffers, self._written = {}, {}, {}
        for modality in streams:
            channels, dtype = MODALITY_FORMAT[modality]
            group = perspective.create_group(modality)
            # Delta only on integer frames: on floats it round-trips exactly or not depending on the values.
            filters = [Delta(dtype=np.dtype(dtype).name)] if np.issubdtype(dtype, np.integer) else None
            self._videos[modality] = group.create_group("video").create_array(
                "data", shape=(channels, n_frames, frame_size, frame_size), dtype=dtype,
                chunks=(channels, self._chunk, frame_size, frame_size),
                compressors=[_COMPRESSOR], filters=filters, fill_value=0)
            group.create_group("timestamps_us").create_array("data", data=timestamps_us)
            self._buffers[modality] = []
            self._written[modality] = 0
        return trace

    def frame(self, rgb: np.ndarray, ir: np.ndarray | None, depth_mm: np.ndarray | None) -> None:
        planes = {"rgb": rgb, "ir": ir, "depth": None if depth_mm is None else depth_mm.astype(np.float32)}
        for modality, buffer in self._buffers.items():
            buffer.append(planes[modality])
            if len(buffer) == self._chunk:
                self._flush(modality)

    def _flush(self, modality: str) -> None:
        buffer = self._buffers[modality]
        if not buffer:
            return
        block = np.stack(buffer)                            # (k, H, W, 3) rgb | (k, H, W) ir, depth
        if block.ndim == 3:
            block = block[..., np.newaxis]                  # (k, H, W, 1)
        start = self._written[modality]
        self._videos[modality][:, start:start + len(buffer)] = np.ascontiguousarray(
            block.transpose(3, 0, 1, 2))                    # (C, k, H, W)
        self._written[modality] = start + len(buffer)
        buffer.clear()

    def commit(self, vessel_ids: np.ndarray, meta: dict) -> None:
        for modality in self._buffers:
            self._flush(modality)
            if self._written[modality] != self._n_frames:
                raise RuntimeError(f"{self.partial}: {modality} got {self._written[modality]} of "
                                   f"{self._n_frames} frames")
        frame_times_s = np.arange(self._n_frames) / self._fps
        traces = {name: np.interp(frame_times_s, self._trace[:, 0], self._trace[:, column])
                  for name, column in TRACE_COLUMNS.items()}
        perspective = self._root[PERSPECTIVE]
        for modality in self._buffers:
            for name, values in traces.items():
                trace_group = perspective[modality].create_group(name)
                trace_group.create_array("data", data=values.astype(np.float64))
                trace_group.attrs["units"] = TRACE_UNITS[name]
        ids = self._root.create_array("vessel_ids", data=np.asarray(vessel_ids, dtype=np.uint8))
        ids.attrs["labels"] = VESSEL_LABELS
        params = meta["params"]
        attrs = {
            "participant": self.name,       # a Path name, so always a string, as the contract requires
            "recording": self.name,
            "posture": POSTURE_BY_ANGLE[float(params["trace"]["posture_deg"])],
            "monk_tone": params["appearance"]["monk_tone"],
            "preset": meta["preset"],
            "seed": meta["seed"],
            "synthetic_neck": meta,
        }
        for key, value in attrs.items():
            self._root.attrs[key] = value
        self._root = None
        shutil.rmtree(self.path, ignore_errors=True)
        self.partial.rename(self.path)

    def discard(self) -> None:
        self._root = None
        shutil.rmtree(self.partial, ignore_errors=True)
```

- [ ] **Step 3: Wire `--zarr` through `generate.py`**

Add the import beside the `FolderSink` import:

```python
from .folder_store import FolderSink
from .zarr_store import ZarrSink, check_config
```

In `generate_sample`, replace the signature and the lines up to `try:`:

```python
def generate_sample(config: GeneratorConfig, seed: int, out_dir: Path, preset: str = "custom", *,
                    zarr: bool = False) -> dict:
    """Render one sample into `out_dir` (or, with `zarr`, the store `{out_dir}.zarr`) and return its metadata
    dict. Guarantees a testable green-channel pulse: both vessels are checked and, if either is under
    MIN_VISIBLE_SNR, the sample is redrawn from an independent sub-stream of `seed`, up to
    MAX_VISIBILITY_ATTEMPTS times. On failure nothing is left behind."""
    validate(config)
    if zarr:
        check_config(config)
    sink = ZarrSink(out_dir) if zarr else FolderSink(out_dir)
    try:
```

Replace `_one`:

```python
def _one(args) -> tuple[int, Exception | None]:
    config, index, seed, out_dir, preset, zarr = args
    try:
        generate_sample(config, seed, out_dir, preset, zarr=zarr)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller; the sink already removed its output
        return index, e
```

In `generate_dataset`, replace the signature and the lines up to the `if jobs <= 1:` line:

```python
def generate_dataset(config: GeneratorConfig, out_root: Path, n: int, start: int, base_seed: int,
                     preset: str, overrides: list[str], jobs: int = 1, *,
                     zarr: bool = False) -> list[tuple[int, Exception | None]]:
    """Generate samples start..start+n-1 (seed = base_seed + i) and write dataset.json. With `zarr`, each
    sample is a store `{i}.zarr` in `out_root` instead of a folder."""
    validate(config)
    if zarr:
        check_config(config)
    else:
        require_ffmpeg()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_args = [(config, i, base_seed + i, out_root / str(i), preset, zarr) for i in range(start, start + n)]
```

In the `index` dict, add `"format"` after `"n": n,`:

```python
        "n": n,
        "format": "zarr" if zarr else "folder",
```

- [ ] **Step 4: Add the flag in `cli.py`**

After `g.add_argument("--jobs", type=int, default=1)`:

```python
    g.add_argument("--zarr", action="store_true",
                   help="write one cache-contract zarr store per sample ({i}.zarr) instead of the folder layout")
```

In `main`, replace the `generate_dataset(...)` call:

```python
            results = generate_dataset(cfg, args.out, n=args.n, start=args.start, base_seed=args.seed,
                                       preset=args.preset, overrides=args.sets, jobs=args.jobs, zarr=args.zarr)
```

- [ ] **Step 5: Document `--zarr` in `README.md`**

Insert after the paragraph ending "The dataset root gets `dataset.json`.":

```markdown
### Zarr stores for remote-physiology

    uv run synthetic-neck generate --zarr --preset neckflix --n 20 --out data/synthetic_zarr

`--zarr` writes one `{i}.zarr` store per sample instead of the folder, in the layout of remote-physiology's cache
contract (`docs/cache-contract.md` there). Perspective `1` holds `rgb`, plus `ir` (uint8) and `depth` (float32 mm)
when the sample has them. Each modality carries per-frame timestamps and ABP/CVP in mmHg interpolated to the
frames. Root attrs hold `participant` (the sample index), `posture`, `monk_tone`, `preset`, `seed` and the full
metadata under `synthetic_neck`; `vessel_ids` is a root array. ffmpeg is not needed, and `trace.posture_deg`
must stay within 0, 45 and 90. Design: `docs/superpowers/specs/2026-09-15-zarr-output-design.md`.
```

- [ ] **Step 6: Generate a small mixed cache**

From the remote-physiology root:

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset lesson --n 4 --set video.frame_size=64 --set streams.depth_ir_probability=0.5 \
    --out "$SCRATCH/zarr"
ls "$SCRATCH/zarr"
for s in "$SCRATCH"/zarr/*.zarr; do echo "$(basename "$s"): $(ls "$s/1" | tr '\n' ' ')"; done
```

Expected: `wrote 4/4 samples`; `ls` shows `1.zarr 2.zarr 3.zarr 4.zarr
dataset.json` and nothing ending in `.partial`. The per-store listing shows at
least one `depth ir rgb` and at least one `rgb` only. If every store has the
same shape, delete `$SCRATCH/zarr` and re-run into it with `--seed 2027`,
`2028`, ... until both appear. Step 7 then validates that directory.

- [ ] **Step 7: Validate**

```bash
uv run python tools/validate_cache.py "$SCRATCH/zarr"
```

Expected: four `PASS` lines and `4/4 stores pass`, exit 0. For a `FAIL`,
fix `zarr_store.py` using the listed violation, delete `$SCRATCH/zarr`, and
repeat Step 6.

- [ ] **Step 8: Confirm the posture refusal**

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset lesson --n 1 --set trace.posture_deg=30 --out "$SCRATCH/zarr_bad"; echo "exit $?"
ls "$SCRATCH/zarr_bad" 2>&1
```

Expected: `error: --zarr writes posture as supine/recumbent/sitting, ...
this config can draw [30.0]`, then `exit 2`, and `ls` reports no such file or
directory.

- [ ] **Step 9: Re-check folder mode against the Task 1 baseline**

```bash
rm -rf "$SCRATCH/folder_after"
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate \
    --preset lesson --n 4 --set video.frame_size=64 --set streams.depth_ir_probability=0.5 \
    --out "$SCRATCH/folder_after"
```

Then run Task 1 Step 5's comparison loop and `diff` again. Expected:
`IDENTICAL`.

- [ ] **Step 10: Commit (in this repository)**

```bash
git -C tools/synthetic_datasets/synthetic_neck add pyproject.toml uv.lock README.md \
    src/synthetic_neck/zarr_store.py src/synthetic_neck/generate.py src/synthetic_neck/cli.py
git -C tools/synthetic_datasets/synthetic_neck commit -F - <<'EOF'
Add generate --zarr: one cache-contract zarr store per sample

With --zarr each sample is written as {i}.zarr in remote-physiology's cache
contract layout instead of the folder: perspective 1 with rgb, plus ir and
float32-mm depth when drawn, per-frame timestamps, and ABP/CVP interpolated
to the frames in mmHg. Root attrs carry participant (the sample index),
posture in Neckflix's vocabulary, monk_tone, preset, seed and the full
metadata; vessel_ids is a root array. Frames stream into a .zarr.partial
store renamed into place after the SNR check, so a visible store is always
complete. ffmpeg is not needed in this mode. A small mixed cache passes
remote-physiology's tools/validate_cache.py.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: remote-physiology: dataset config, cache spec, submodule pointer

**Files (all in remote-physiology):**

- Create: `configs/datasets/synthetic_neck.yaml`
- Create: `dataset/data_loader/SYNTHETIC_NECK.md`
- Modify: the `tools/synthetic_datasets/synthetic_neck` gitlink

**Interfaces:**

- Consumes: the store layout and attrs Task 2 writes; remote-physiology's
  `--test-participant-dataset` / `--test-participant-id` flags, where the
  dataset name is the config file's stem.
- Produces: the dataset name `synthetic_neck`.

- [ ] **Step 1: Create `configs/datasets/synthetic_neck.yaml`**

```yaml
# Synthetic neck: which stores participate. Nothing here about channels,
# traces, windows or rates — that is the interface's business.
#
# The stores are written by `synthetic-neck generate --zarr` (the submodule at
# tools/synthetic_datasets/synthetic_neck); dataset/data_loader/SYNTHETIC_NECK.md
# is their cache spec. FILTERS is keyed by the store's own root attrs, exactly
# as the generator writes them. Every entry carries both an `include` and an
# `exclude` list; an empty list is no constraint on that side. A store is
# admitted when it passes every entry.
#
# Values compare as strings. A store lacking an attr (or carrying null) fails
# an include on it and passes an exclude. Nested attrs are addressed with dots
# (`synthetic_neck.params.trace.heart_rate_bpm`). `FILTERS: {}` admits every
# store.
#
# Filterable root attrs (one store = one sample, perspective "1"; rgb, plus ir
# and depth when the sample drew them):
#   participant        '1', '2', ...; the sample index. Ids collide with other
#                      datasets' ids, so hold one out with
#                      --test-participant-dataset synthetic_neck
#   recording          same value as participant
#   posture            supine | recumbent | sitting (Neckflix's spelling)
#   monk_tone          Monk skin tone 1..10, or null when not drawn
#   preset             lesson | benchmark | neckflix | custom
#   seed               the sample's seed (int)
#   synthetic_neck.*   the generator's full per-sample metadata: seed, preset,
#                      version, frame_size, fps, n_frames, streams, config.*,
#                      params.* (the drawn values), geometry_px.*, derived.*,
#                      visibility.* (green-channel SNR and draw attempts)
CACHED_PATH: "D:/synthetic_neck_zarr"
FILTERS: {}
```

- [ ] **Step 2: Create `dataset/data_loader/SYNTHETIC_NECK.md`**

````markdown
# Synthetic Neck Cache Spec

Synthetic neck videos with ground-truth arterial (ABP) and central venous
(CVP) pressure, rendered by `synthetic-neck`, the generator carried as a
submodule at `tools/synthetic_datasets/synthetic_neck`. There is no raw
dataset and no separate cacher: `synthetic-neck generate --zarr` renders
straight into stores that satisfy [the cache contract](../../docs/cache-contract.md).
The validator is the acceptance test. Generator design:
`tools/synthetic_datasets/synthetic_neck/docs/superpowers/specs/2026-09-15-zarr-output-design.md`.

## Building

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset neckflix --n 200 --jobs 8 --out <cache dir>
uv run python tools/validate_cache.py <cache dir>
```

Presets: `lesson` (large pulse, flat lighting), `benchmark` (moderate pulse,
controlled degradations), `neckflix` (ranges calibrated from Neckflix).
`--set block.field=value` overrides any generator field. `--seed` is the base
seed (sample `i` uses `seed + i`), so a cache can be rebuilt from its
`dataset.json`.

## What the generator writes

```text
{out}/
|-- dataset.json                    preset, overrides, seeds, generator version and commit, "format": "zarr"
`-- 1.zarr                          one store per sample; the name is the sample index
    |-- attrs                       see "Root attributes"
    |-- vessel_ids   (H, W) uint8   0 background, 1 artery, 2 vein; attrs: labels
    `-- 1/                          attrs: fps = video.fps (30 by default)
        |-- rgb/
        |   |-- timestamps_us/data  (T,) int64          round(i * 1e6 / fps)
        |   |-- video/data          (3, T, H, W) uint8   Delta + blosc-zstd
        |   |-- abp/data            (T,) float64        attrs: units="mmHg"
        |   `-- cvp/data            (T,) float64        attrs: units="mmHg"
        |-- ir/                     only when the sample drew IR + depth
        |   `-- video/data          (1, T, H, W) uint8, plus rgb's three siblings
        `-- depth/                  only when the sample drew IR + depth
            `-- video/data          (1, T, H, W) float32 mm, blosc-zstd without Delta, plus rgb's three siblings
```

- **Frames**: rendered square at `video.frame_size` (300 by default), with
  no resize at write time; resizing is consumer-side as usual. Chunks are
  `(C, min(32, T), H, W)`.
- **Timestamps**: synthesised from the nominal rate, starting at 0 and
  identical in every modality.
- **`abp`, `cvp`**: the generator's 1 kHz traces, linearly interpolated at
  the frame times. They are exactly the traces the pixels were rendered from,
  and the same arrays sit under every modality.
- **`depth`**: float32 millimetres. That is Neckflix's unit but not its
  integer dtype, so the pulse's 0.3-0.5 mm skin lift survives. Kinect-like
  noise (1.6 mm sd at 1 m, growing with distance squared) is already in it.
- **`ir`**: uint8, not Neckflix's uint16 Kinect IR; the two scales are not
  comparable.
- **`vessel_ids`**: one static map per sample (nothing in the scene moves),
  on the frames' pixel grid. It is not part of the contract, and the
  validator and the reader only walk root groups, so neither sees it. If
  frames are resized consumer-side, resize this map nearest-neighbour to
  match.
- **Guaranteed pulse**: the generator redraws a sample, up to 20 times,
  until the green channel's vessel-averaged cardiac SNR is at least 10 for
  both vessels. IR and depth carry no such guarantee.

### Root attributes

```python
{
  "participant": "1",          # the sample index; every sample is its own participant
  "recording": "1",            # same value
  "posture": "recumbent",      # supine | recumbent | sitting, from trace.posture_deg 0 | 45 | 90
  "monk_tone": 6,              # Monk skin tone 1-10 when the preset draws one, else null
  "preset": "neckflix",
  "seed": 2027,                # this sample's seed: base seed 2026 + index 1
  "synthetic_neck": {...},     # full per-sample metadata: config, drawn params, geometry_px,
                               # derived quantities, visibility (SNR, attempts)
}
```

Configs filter on these as written, e.g.
`posture: {include: [supine], exclude: []}`. Drawn values are reachable by
dotted path (`synthetic_neck.params.trace.heart_rate_bpm`), but filters match
exact strings, so they select exact values, not ranges.

## Mixing with other datasets

The ids `"1"`, `"2"`, ... collide with other datasets' ids. Hold out a
participant together with its dataset:
`--test-participant-dataset synthetic_neck --test-participant-id 3`. The
dataset name is the stem of `configs/datasets/synthetic_neck.yaml`.

## Liberties taken

- Timestamps are synthesised, not measured.
- Traces are interpolated from 1 kHz onto the frame grid.
- Depth is float32 rather than a sensor's integer millimetres.
- `vessel_ids` is an uncontracted root array.
- The `synthetic_neck` metadata keeps the generator's folder-layout fields
  verbatim (`vessel_ids.file`, `depth_units_mm`, `rgbid_streams`). Inside a
  store they describe the folder layout, not the store.
````

- [ ] **Step 3: Check the submodule is at the Task 2 commit**

```bash
git -C tools/synthetic_datasets/synthetic_neck log --oneline -3
git diff --submodule tools/synthetic_datasets/synthetic_neck
```

Expected: the log's top entry is "Add generate --zarr: ...", and the diff
lists the spec, plan, Task 1 and Task 2 commits since `1a207a9`.

- [ ] **Step 4: Commit (in remote-physiology)**

```bash
git add configs/datasets/synthetic_neck.yaml dataset/data_loader/SYNTHETIC_NECK.md tools/synthetic_datasets/synthetic_neck
git commit -F - <<'EOF'
feat: synthetic-neck zarr caches as a dataset

`synthetic-neck generate --zarr` now renders straight into cache-contract
stores, so synthetic data is a dataset like PURE or Neckflix:
configs/datasets/synthetic_neck.yaml names it (and the
--test-participant-dataset label), and dataset/data_loader/SYNTHETIC_NECK.md
is its cache spec. The submodule moves to the commit that adds --zarr.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

Do not push either repository without asking. Until synthetic_neck's
`feat/zarr-output` is pushed, this gitlink does not resolve in other clones.
