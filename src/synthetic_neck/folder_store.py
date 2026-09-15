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
