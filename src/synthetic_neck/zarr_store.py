"""Per-sample zarr output: one store satisfying remote-physiology's cache contract (docs/cache-contract.md there).

    {i}.zarr                  root attrs: participant, recording, posture, abp_site, monk_tone, preset, seed,
                              synthetic_neck
    |-- vessel_ids            (H, W) uint8, attrs: labels
    `-- 1/                    attrs: fps
        `-- rgb | ir | depth  video/data (C, T, H, W); timestamps_us/data (T,) int64;
                              abp, cvp (mmHg), ecg (mV), ppg, rr (arb): <trace>/data (T,) float64, attrs units

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
TRACE_COLUMNS = {"abp": 1, "cvp": 2, "ecg": 3, "ppg": 4, "rr": 5}          # columns of generate_trace's array
TRACE_UNITS = {"abp": "mmHg", "cvp": "mmHg", "ecg": "mV", "ppg": "arb", "rr": "arb"}
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
            "abp_site": params["trace"]["abp_site"],
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
        shutil.rmtree(self.path, ignore_errors=True)
