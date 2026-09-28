"""Per-sample zarr output: one store in the layout of remote-physiology's cache contract.

    {i}.zarr                  root attrs: participant, recording, posture, abp_site, monk_tone, seed, synthetic_neck
    |-- vessel_ids            (H, W) uint8, attrs: labels (0 background, 1 artery, 2 vein)
    |-- neck_mask             (H, W) uint8, 1 on the neck
    `-- 1/                    attrs: fps
        `-- rgb | ir | depth  video/data (C, T, H, W): rgb uint8, ir uint16, depth uint16 integer mm;
                              timestamps_us/data (T,) int64; abp, cvp (mmHg), ecg (mV), ppg, rr (arb):
                              <trace>/data (T,) float64 on the frame clock, attrs units

Frames stream into `{i}.zarr.partial` a chunk at a time and `commit` renames it into place, so a visible store is
complete.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

from .sample import TRACE_UNITS

PERSPECTIVE = "1"
CHUNK_FRAMES = 32
VESSEL_LABELS = {"0": "background", "1": "artery", "2": "vein"}
MODALITY_FORMAT = {"rgb": (3, np.uint8), "ir": (1, np.uint16), "depth": (1, np.uint16)}   # (channels, dtype)
_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")


class ZarrSink:
    """Writes one sample as `{out_dir}.zarr`."""

    def __init__(self, out_dir: Path):
        out_dir = Path(out_dir)
        self.name = out_dir.name
        self.path = out_dir.with_name(f"{self.name}.zarr")
        self.partial = out_dir.with_name(f"{self.name}.zarr.partial")
        self._root = None
        self._videos, self._buffers, self._written = {}, {}, {}
        self._n_frames, self._chunk = 0, CHUNK_FRAMES

    def begin(self, n_frames: int, frame_px: int, fps: float) -> None:
        shutil.rmtree(self.partial, ignore_errors=True)
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(str(self.partial), mode="w")
        self._n_frames, self._chunk = n_frames, min(CHUNK_FRAMES, n_frames)
        perspective = self._root.create_group(PERSPECTIVE)
        perspective.attrs["fps"] = float(fps)
        timestamps_us = np.round(np.arange(n_frames) * 1e6 / fps).astype(np.int64)
        for modality, (channels, dtype) in MODALITY_FORMAT.items():
            group = perspective.create_group(modality)
            self._videos[modality] = group.create_group("video").create_array(
                "data", shape=(channels, n_frames, frame_px, frame_px), dtype=dtype,
                chunks=(channels, self._chunk, frame_px, frame_px),
                compressors=[_COMPRESSOR], filters=[Delta(dtype=np.dtype(dtype).name)], fill_value=0)
            group.create_group("timestamps_us").create_array("data", data=timestamps_us)
            self._buffers[modality], self._written[modality] = [], 0

    def frame(self, rgb: np.ndarray, ir: np.ndarray, depth_mm: np.ndarray) -> None:
        for modality, plane in (("rgb", rgb), ("ir", ir), ("depth", depth_mm)):
            self._buffers[modality].append(plane)
            if len(self._buffers[modality]) == self._chunk:
                self._flush(modality)

    def _flush(self, modality: str) -> None:
        buffer = self._buffers[modality]
        if not buffer:
            return
        block = np.stack(buffer)                            # (k, H, W, 3) or (k, H, W)
        if block.ndim == 3:
            block = block[..., np.newaxis]
        start = self._written[modality]
        self._videos[modality][:, start:start + len(buffer)] = np.ascontiguousarray(block.transpose(3, 0, 1, 2))
        self._written[modality] = start + len(buffer)
        buffer.clear()

    def commit(self, traces: dict[str, np.ndarray], labels: np.ndarray, neck_mask: np.ndarray, meta: dict) -> None:
        for modality in self._buffers:
            self._flush(modality)
            if self._written[modality] != self._n_frames:
                raise RuntimeError(f"{self.partial}: {modality} got {self._written[modality]} of {self._n_frames} frames")
        perspective = self._root[PERSPECTIVE]
        for modality in self._buffers:
            for name, values in traces.items():
                if values.shape != (self._n_frames,):
                    raise RuntimeError(f"{self.partial}: trace {name} has {values.shape}, frames {self._n_frames}")
                group = perspective[modality].create_group(name)
                group.create_array("data", data=np.asarray(values, dtype=np.float64))
                group.attrs["units"] = TRACE_UNITS[name]
        ids = self._root.create_array("vessel_ids", data=np.asarray(labels, dtype=np.uint8))
        ids.attrs["labels"] = VESSEL_LABELS
        self._root.create_array("neck_mask", data=np.asarray(neck_mask, dtype=np.uint8))
        for key, value in {"participant": self.name, "recording": self.name, "posture": meta["posture"],
                           "abp_site": meta["abp_site"], "monk_tone": meta["appearance"]["monk_tone"],
                           "seed": meta["seed"], "synthetic_neck": meta}.items():
            self._root.attrs[key] = value
        self._root = None
        shutil.rmtree(self.path, ignore_errors=True)
        self.partial.rename(self.path)

    def discard(self) -> None:
        self._root = None
        shutil.rmtree(self.partial, ignore_errors=True)
        shutil.rmtree(self.path, ignore_errors=True)
