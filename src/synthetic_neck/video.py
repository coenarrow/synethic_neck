"""Lossless MKV writing via an ffmpeg subprocess (FFV1 codec, VLC-playable)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

# name -> (raw pix_fmt fed to ffmpeg, FFV1 storage pix_fmt, numpy dtype, channels)
FORMATS = {
    "gray": ("gray", "gray", np.uint8, 1),
    "rgb": ("rgb24", "bgr0", np.uint8, 3),      # bgr0 = lossless packed RGB
    "gray16": ("gray16le", "gray16le", np.uint16, 1),
}


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH")


class MkvWriter:
    """Write one video stream of `fmt` frames to an FFV1/MKV file."""

    def __init__(self, path: Path, size: tuple[int, int], fmt: str = "rgb", fps: float = 30.0):
        require_ffmpeg()
        in_fmt, out_fmt, self.dtype, self.channels = FORMATS[fmt]
        self.path, self.size = Path(path), size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = size
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "ffv1", "-level", "3", "-pix_fmt", out_fmt,
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        expected = self.size + ((self.channels,) if self.channels > 1 else ())
        if frame.shape != expected or frame.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype.__name__} {expected}, got {frame.dtype} {frame.shape}")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {err}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def mux(inputs: list[Path], out: Path, titles: list[str] | None = None) -> None:
    """Combine single-stream MKVs into one multi-stream MKV (stream copy)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    for i in range(len(inputs)):
        cmd += ["-map", str(i)]
    for i, t in enumerate(titles or []):
        cmd += [f"-metadata:s:v:{i}", f"title={t}"]
    cmd += ["-c", "copy", str(out)]
    subprocess.run(cmd, check=True)


def read_mkv(path: Path, fmt: str = "rgb", stream: int = 0) -> np.ndarray:
    """Decode one video stream of an MKV into a (T, H, W[, C]) array."""
    in_fmt, _, dtype, channels = FORMATS[fmt]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", f"v:{stream}", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:v:{stream}",
         "-f", "rawvideo", "-pix_fmt", in_fmt, "-"],
        capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=dtype)
    shape = (-1, h, w) + ((channels,) if channels > 1 else ())
    return arr.reshape(shape)
