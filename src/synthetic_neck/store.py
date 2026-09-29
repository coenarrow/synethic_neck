"""Per-sample zarr output: one store in the layout of remote-physiology's cache contract.

    {i}.zarr                  root attrs: participant, recording, posture, abp_site, skin_tone (Monk 1..10),
                              neck_circumference_cm, seed, synthetic_neck
    |-- vessel_ids            (H, W) uint8, attrs: labels (0 background, 1 artery, 2 vein)
    |-- neck_mask             (H, W) uint8, 1 on the neck
    `-- 1/                    attrs: fps
        `-- rgb | ir | depth  video/data (C, T, H, W): rgb uint8, ir uint16, depth uint16 integer mm;
                              timestamps_us/data (T,) int64; abp, cvp (mmHg), ecg (mV), ppg, rr (arb):
                              <trace>/data (T,) float64 on the frame clock, attrs units

Frames stream into `{i}.zarr.partial` a batch at a time (a batch the size of a chunk, CHUNK_FRAMES, writes whole
chunks) and `commit` renames it into place, so a visible store is complete.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

from .frames.sensor import SensedFrames
from .sample import TRACE_UNITS

PERSPECTIVE = "1"
CHUNK_FRAMES = 32
VESSEL_LABELS = {"0": "background", "1": "artery", "2": "vein"}
MODALITY_FORMAT = {"rgb": (3, np.uint8), "ir": (1, np.uint16), "depth": (1, np.uint16)}   # (channels, dtype)
# zstd 5 by default: the sensor noise makes the frames barely compressible, so level 9 buys 8% over level 5 at 18
# times the write time, which at 9 was the whole cost of a sample.
DEFAULT_CLEVEL = 5
RENAME_WAIT_S = 10.0


def compressor(clevel: int = DEFAULT_CLEVEL) -> BloscCodec:
    """The video codec at zstd level `clevel`, 1 (fastest) to 9 (smallest); 0 stores the frames uncompressed."""
    if not 0 <= clevel <= 9:
        raise ValueError(f"compression level must be 0..9, got {clevel}")
    return BloscCodec(cname="zstd", clevel=clevel, shuffle="bitshuffle")


def rename_when_released(source: Path, target: Path) -> None:
    """Rename a directory, waiting out RENAME_WAIT_S of PermissionError. Windows refuses the rename while any
    process has a file inside open, and a virus scanner or indexer opens the files just written for up to a few
    hundred milliseconds."""
    deadline = time.monotonic() + RENAME_WAIT_S
    while True:
        try:
            source.rename(target)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


class ZarrSink:
    """Writes one sample as `{out_dir}.zarr`."""

    def __init__(self, out_dir: Path, clevel: int = DEFAULT_CLEVEL):
        out_dir = Path(out_dir)
        self.name = out_dir.name
        self.compressor = compressor(clevel)
        self.path = out_dir.with_name(f"{self.name}.zarr")
        self.partial = out_dir.with_name(f"{self.name}.zarr.partial")
        self._root = None
        self._videos, self._written = {}, {}
        self._n_frames = 0

    def begin(self, n_frames: int, frame_px: int, fps: float) -> None:
        shutil.rmtree(self.partial, ignore_errors=True)
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(str(self.partial), mode="w")
        self._n_frames = n_frames
        chunk = min(CHUNK_FRAMES, n_frames)
        perspective = self._root.create_group(PERSPECTIVE)
        perspective.attrs["fps"] = float(fps)
        timestamps_us = np.round(np.arange(n_frames) * 1e6 / fps).astype(np.int64)
        for modality, (channels, dtype) in MODALITY_FORMAT.items():
            group = perspective.create_group(modality)
            self._videos[modality] = group.create_group("video").create_array(
                "data", shape=(channels, n_frames, frame_px, frame_px), dtype=dtype,
                chunks=(channels, chunk, frame_px, frame_px),
                compressors=[self.compressor], filters=[Delta(dtype=np.dtype(dtype).name)], fill_value=0)
            group.create_group("timestamps_us").create_array("data", data=timestamps_us)
            self._written[modality] = 0

    def frames(self, batch: SensedFrames) -> None:
        """Append a batch: (B, 3, H, W) colour, (B, H, W) infrared and depth, as the next B frames."""
        for modality, block in (("rgb", batch.rgb), ("ir", batch.ir[:, None]), ("depth", batch.depth_mm[:, None])):
            start = self._written[modality]
            self._videos[modality][:, start:start + len(block)] = np.ascontiguousarray(block.transpose(1, 0, 2, 3))
            self._written[modality] = start + len(block)

    def commit(self, traces: dict[str, np.ndarray], labels: np.ndarray, neck_mask: np.ndarray, meta: dict) -> None:
        for modality in self._written:
            if self._written[modality] != self._n_frames:
                raise RuntimeError(f"{self.partial}: {modality} got {self._written[modality]} of {self._n_frames} frames")
        perspective = self._root[PERSPECTIVE]
        for modality in self._written:
            for name, values in traces.items():
                if values.shape != (self._n_frames,):
                    raise RuntimeError(f"{self.partial}: trace {name} has {values.shape}, frames {self._n_frames}")
                group = perspective[modality].create_group(name)
                group.create_array("data", data=np.asarray(values, dtype=np.float64))
                group.attrs["units"] = TRACE_UNITS[name]
        ids = self._root.create_array("vessel_ids", data=np.asarray(labels, dtype=np.uint8))
        ids.attrs["labels"] = VESSEL_LABELS
        self._root.create_array("neck_mask", data=np.asarray(neck_mask, dtype=np.uint8))
        neck_circumference_cm = 2 * np.pi * meta["scene"]["neck_radius_mm"] / 10.0
        for key, value in {"participant": self.name, "recording": self.name, "posture": meta["posture"],
                           "abp_site": meta["abp_site"], "skin_tone": meta["appearance"]["monk_tone"],
                           "neck_circumference_cm": float(neck_circumference_cm),
                           "seed": meta["seed"], "synthetic_neck": meta}.items():
            self._root.attrs[key] = value
        self._root = None
        shutil.rmtree(self.path, ignore_errors=True)
        rename_when_released(self.partial, self.path)

    def discard(self) -> None:
        self._root = None
        shutil.rmtree(self.partial, ignore_errors=True)
        shutil.rmtree(self.path, ignore_errors=True)
