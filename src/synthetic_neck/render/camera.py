"""Sensor stage: blur, area downsample, signal-dependent noise, quantisation."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import CameraParams, SensorParams

DEPTH_UNITS_MM = 0.02  # 16-bit depth stores depth_mm / DEPTH_UNITS_MM (range 0-1310 mm)


def area_resample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix that box-averages n_in samples into n_out bins."""
    m = np.zeros((n_out, n_in))
    edges = np.linspace(0, n_in, n_out + 1)
    for i in range(n_out):
        lo, hi = edges[i], edges[i + 1]
        for j in range(int(np.floor(lo)), int(np.ceil(hi))):
            m[i, j] = min(hi, j + 1) - max(lo, j)
    return m / (n_in / n_out)


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    padded = np.pad(img, pad, mode="edge")
    windows = sliding_window_view(padded, len(k), axis=axis)
    return windows @ k


def gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur over the first two axes (H, W[, C]); edge-padded."""
    if sigma_px <= 0:
        return img
    k = _gauss_kernel(sigma_px)
    return _convolve_axis(_convolve_axis(img, k, 0), k, 1)


def depth_to_uint16(depth_mm: np.ndarray) -> np.ndarray:
    q = np.rint(depth_mm / DEPTH_UNITS_MM)
    if q.max() > 65535 or q.min() < 0:
        raise ValueError(f"depth {depth_mm.min():.0f}-{depth_mm.max():.0f} mm exceeds 16-bit range")
    return q.astype(np.uint16)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma, uint8 (H, W)."""
    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


class SensorStage:
    """Turns a crop-resolution float image into an output-resolution sensor frame."""

    def __init__(self, params: SensorParams, camera: CameraParams, out_size: int):
        self.p = params
        self.camera = camera
        self._down = area_resample_matrix(camera.crop_px, out_size)
        self.rng = np.random.default_rng(params.seed)
        # Kinect-like depth noise grows with the square of distance.
        self.depth_noise_sd_mm = params.depth_noise_mm_at_1m * (camera.distance_mm / 1000.0) ** 2

    def downsample(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return np.stack([self.downsample(img[..., c]) for c in range(img.shape[-1])], axis=-1)
        return self._down @ img @ self._down.T

    def _noise(self, signal: np.ndarray, read_sd: float) -> np.ndarray:
        var = read_sd ** 2 + self.p.shot_noise_gain * np.clip(signal, 0, None)
        return self.rng.standard_normal(signal.shape) * np.sqrt(var)

    def _quantise8(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    def rgb(self, img: np.ndarray, noise: bool = True) -> np.ndarray:
        img = self.downsample(gaussian_blur(img, self.p.blur_sigma_px))
        if noise:
            img = img + self._noise(img, self.p.read_noise_sd)
        return self._quantise8(img)

    def ir(self, img: np.ndarray, noise: bool = True) -> np.ndarray:
        img = self.downsample(gaussian_blur(img, self.p.blur_sigma_px))
        if noise:
            img = img + self._noise(img, self.p.ir_read_noise_sd)
        return self._quantise8(img)

    def depth(self, depth_mm: np.ndarray, noise: bool = True) -> np.ndarray:
        d = self.downsample(depth_mm)
        if noise and self.depth_noise_sd_mm > 0:
            d = d + self.rng.standard_normal(d.shape) * self.depth_noise_sd_mm
        return d
