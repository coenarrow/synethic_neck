"""Static scene: skin colour, texture, vignette, faint vessel darkening, and a
Lambertian cylinder whose axis runs along the vessels (shared with depth)."""
from __future__ import annotations

import numpy as np

from ..config import AppearanceParams, CameraParams
from ..geometry import VesselGeometry


class BaseScene:
    def __init__(self, appearance: AppearanceParams, geometry: VesselGeometry, camera: CameraParams,
                 w_art: np.ndarray, w_vein: np.ndarray):
        a, g, n = appearance, geometry, geometry.frame_size
        rng = np.random.default_rng(a.seed)
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
        r2 = ((xs - n / 2) ** 2 + (ys - n / 2) ** 2) / (n / 2) ** 2
        texture = rng.standard_normal((n, n)) * a.texture_sd

        # Cylinder: axis through the frame centre along the vessel direction.
        pixel_scale_mm = camera.native_scale_mm            # mm per rendered pixel
        self.perp_mm = ((xs - n / 2) * g.normal[0] + (ys - n / 2) * g.normal[1]) * pixel_scale_mm
        r = a.neck_radius_mm
        cos_theta = np.sqrt(np.clip(1.0 - (self.perp_mm / r) ** 2, 0.0, 1.0))
        self.shading = 1.0 - a.shading_strength * (1.0 - cos_theta)
        self.depth_mm = camera.distance_mm - np.sqrt(np.clip(r ** 2 - self.perp_mm ** 2, 0, None))

        static = -a.vignette * r2 + texture - a.static_vessel_contrast * (w_art + w_vein)
        skin = np.asarray(a.skin_rgb, dtype=np.float64)
        self.rgb = skin[None, None, :] * self.shading[..., None] + static[..., None]
        self.ir = (a.ir_base * self.shading + 0.5 * texture - 0.5 * a.vignette * r2
                   - a.ir_vein_contrast * w_vein - a.static_vessel_contrast * w_art)
