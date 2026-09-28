"""Sensor: the continuous frame becomes what the camera records. In order: a Gaussian blur on the native grid for the
lens and demosaic, an exact area average down to the delivered size, Gaussian noise per pixel and frame at that size,
and quantisation to the delivered types: colour to uint8, infrared to uint16, depth to uint16 integer millimetres by
stochastic rounding so that a lift smaller than a millimetre still moves its share of the pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ..traces.priors import Normal, Range
from .distension import quantise_depth
from .render import Frame
from .resample import area_average, area_weights, noise_gain, resample_labels  # noqa: F401  re-exported


@dataclass(frozen=True)
class SensorConfig:
    blur_sigma_px: Range            # Gaussian blur on the native grid, native pixels
    rgb_noise: Range                # sd per channel, 8-bit levels, at the delivered size
    ir_noise: Normal                # sd, 16-bit units, at the delivered size
    depth_noise_mm: Range           # sd, mm, at the delivered size


@dataclass(frozen=True)
class Sensor:
    blur_sigma_px: float
    rgb_noise: float
    ir_noise: float
    depth_noise_mm: float


def draw_sensor(cfg: SensorConfig, rng: np.random.Generator) -> Sensor:
    return Sensor(cfg.blur_sigma_px.draw(rng), cfg.rgb_noise.draw(rng), cfg.ir_noise.draw(rng),
                  cfg.depth_noise_mm.draw(rng))


@dataclass(frozen=True)
class SensedFrame:
    rgb: np.ndarray                 # (N, N, 3) uint8
    ir: np.ndarray                  # (N, N) uint16
    depth_mm: np.ndarray            # (N, N) uint16, integer millimetres


def blur(x: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return np.asarray(x, dtype=np.float64)
    sig = (sigma_px, sigma_px) + ((0,) if x.ndim == 3 else ())
    return gaussian_filter(np.asarray(x, dtype=np.float64), sig, mode="nearest")


def sense(frame: Frame, sensor: Sensor, output_px: int, rng: np.random.Generator) -> SensedFrame:
    """Record `frame`: blur, area-average to `output_px`, add the drawn noise, quantise."""
    rgb = area_average(blur(frame.rgb, sensor.blur_sigma_px), output_px)
    ir = area_average(blur(frame.ir, sensor.blur_sigma_px), output_px)
    depth = area_average(blur(frame.depth_mm, sensor.blur_sigma_px), output_px)
    rgb = rgb + sensor.rgb_noise * rng.standard_normal(rgb.shape)
    ir = ir + sensor.ir_noise * rng.standard_normal(ir.shape)
    depth = depth + sensor.depth_noise_mm * rng.standard_normal(depth.shape)
    return SensedFrame(np.clip(np.rint(rgb), 0, 255).astype(np.uint8),
                       np.clip(np.rint(ir), 0, 65535).astype(np.uint16),
                       quantise_depth(depth, rng))
