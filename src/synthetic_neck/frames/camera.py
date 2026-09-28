"""Pinhole camera. The rig is fixed (field of view, native width, crop size); the distance to the skin at the frame
centre is drawn per sample. The delivered size defaults to the native crop; a smaller one is an argument at
generation time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.priors import Range


@dataclass(frozen=True)
class CameraConfig:
    distance_mm: Range          # camera to the skin surface at the frame centre
    native_width_px: float      # sensor width in pixels
    hfov_deg: float             # horizontal field of view
    crop_px: float              # side of the square crop rendered, in native pixels


@dataclass(frozen=True)
class Camera:
    distance_mm: float
    focal_px: float             # pinhole focal length in native pixels
    crop_px: int
    output_px: int

    @property
    def pixel_scale_mm(self) -> float:
        """mm per native pixel on a plane at the centre distance."""
        return self.distance_mm / self.focal_px

    def rays(self, n: int) -> np.ndarray:
        """(n, n, 3) ray directions with unit z for an n-pixel grid covering the crop, pixel centres, camera
        coordinates x right, y down, z forward. n = crop_px renders at native resolution; n = output_px gives
        one ray per delivered pixel."""
        c = (np.arange(n) + 0.5) / n - 0.5                      # fraction of the crop from its centre
        u = c * self.crop_px / self.focal_px                    # tan of the angle off axis
        ux, uy = np.meshgrid(u, u)
        return np.stack([ux, uy, np.ones_like(ux)], axis=-1)


def draw_camera(cfg: CameraConfig, rng: np.random.Generator, output_px: int | None = None) -> Camera:
    """`output_px` is the delivered frame size; None delivers the crop at native resolution."""
    focal = 0.5 * cfg.native_width_px / np.tan(np.deg2rad(cfg.hfov_deg) / 2)
    crop = int(round(cfg.crop_px))
    return Camera(cfg.distance_mm.draw(rng), float(focal), crop, crop if output_px is None else int(output_px))
