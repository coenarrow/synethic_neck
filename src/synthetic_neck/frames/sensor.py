"""Sensor: the continuous frames become what the camera records. In order: a Gaussian blur on the native grid for the
lens and demosaic, an exact area average down to the delivered size, Gaussian noise per pixel and frame at that size,
and quantisation to the delivered types: colour to uint8, infrared to uint16, depth to uint16 integer millimetres by
stochastic rounding so that a lift smaller than a millimetre still moves its share of the pixels.

A batch of frames is recorded together, its five planes (three colour, infrared, depth) as one (B, 5, n, n) array.
The blur and the area average are both linear and separable, so they are one (N, n) matrix M per axis, built once per
sample, and the batch is recorded as M x M^T. The noise is drawn in that order: the Gaussian noise of the whole batch,
then the uniform draws of the depth rounding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.priors import Normal, Range
from .backend import Backend
from .render import Frames
from .resample import area_average, area_weights, noise_gain, resample_labels, sensor_matrix  # noqa: F401  re-exported


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
class SensedFrames:
    """A batch of recorded frames, channel first as the store keeps them."""
    rgb: np.ndarray                 # (B, 3, N, N) uint8
    ir: np.ndarray                  # (B, N, N) uint16
    depth_mm: np.ndarray            # (B, N, N) uint16, integer millimetres

    def __len__(self) -> int:
        return self.rgb.shape[0]


class Recorder:
    """The drawn sensor on a device: the matrix of its linear stage and the noise levels, ready for any batch."""

    def __init__(self, sensor: Sensor, native_px: int, output_px: int, backend: Backend):
        self.backend, self.sensor = backend, sensor
        self.matrix = backend.array(sensor_matrix(native_px, output_px, sensor.blur_sigma_px))    # (N, n)
        self.noise_sd = backend.array([sensor.rgb_noise] * 3 + [sensor.ir_noise, sensor.depth_noise_mm])[None, :, None, None]

    def sense(self, frames: Frames, rng: np.random.Generator) -> SensedFrames:
        """Record `frames`: blur and area-average by the matrix, add the drawn noise, quantise."""
        bk, xp, m = self.backend, self.backend.xp, self.matrix
        planes = xp.concatenate([frames.rgb, frames.ir[:, None], frames.depth_mm[:, None]], 1)   # (B, 5, n, n)
        planes = m @ planes @ m.T
        planes = planes + self.noise_sd * bk.normal(tuple(planes.shape), rng)
        rgb = xp.clip(xp.round(planes[:, :3]), 0, 255)
        ir = xp.clip(xp.round(planes[:, 3]), 0, 65535)
        depth = quantise_depth(planes[:, 4], rng, bk)
        return SensedFrames(bk.numpy(rgb).astype(np.uint8), bk.numpy(ir).astype(np.uint16),
                            bk.numpy(depth).astype(np.uint16))


def quantise_depth(depth_mm, rng: np.random.Generator, backend: Backend):
    """Depth in integer millimetres by stochastic rounding: a value of 700.3 is 701 with probability 0.3 and 700
    otherwise, so a lift smaller than a millimetre still moves that fraction of the pixels. Values in [0, 65535]."""
    xp = backend.xp
    return xp.clip(xp.floor(depth_mm + backend.uniform(tuple(depth_mm.shape), rng)), 0, 65535)
