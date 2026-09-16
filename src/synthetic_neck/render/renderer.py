"""Compose the stages: base scene -> pulse -> illumination -> sensor."""
from __future__ import annotations

import numpy as np

from ..config import SampleParams
from ..geometry import id_map, scaled, tube_fields, vessel_geometry
from .base import BaseScene
from .camera import SensorStage
from .illumination import IlluminationStage
from .pulse import PulseStage


class Renderer:
    def __init__(self, params: SampleParams, trace: np.ndarray):
        self.params = params
        self.fps = params.video.fps
        self.n_frames = params.video.n_frames
        self.geometry = vessel_geometry(params.geometry, params.camera)          # output px
        self.render_geometry = scaled(self.geometry, params.camera.render_scale)  # crop px
        self.pixel_scale_mm = params.camera.native_scale_mm                       # mm per rendered px
        self.ids = id_map(self.geometry)

        g = self.render_geometry
        w_art, _ = tube_fields(g, "artery")
        w_vein, _ = tube_fields(g, "vein")
        self.base = BaseScene(params.appearance, g, params.camera, w_art, w_vein)
        self.pulse = PulseStage(params.pulse, params.geometry, g, self.pixel_scale_mm, trace,
                                site_delay_s=params.trace.abp_site_delay_s)
        self.illumination = IlluminationStage(params.illumination, g.frame_size, self.fps, self.n_frames,
                                              self.base.shading)
        self.sensor = SensorStage(params.sensor, params.camera, params.video.frame_size)

    def frame_time(self, i: int) -> float:
        return i / self.fps

    def frame(self, i: int, noise: bool = True) -> np.ndarray:
        """RGB uint8 (H, W, 3) at output resolution."""
        img = self.base.rgb + self.pulse.rgb_mod(self.frame_time(i))
        img = self.illumination(img, i)
        return self.sensor.rgb(img, noise=noise)

    def ir_frame(self, i: int, noise: bool = True) -> np.ndarray:
        """Near-infrared uint8 (H, W)."""
        img = self.base.ir + self.pulse.ir_mod(self.frame_time(i))
        img = self.illumination(img, i)
        return self.sensor.ir(img, noise=noise)

    def depth_frame(self, i: int, noise: bool = True) -> np.ndarray:
        """Depth in mm, float64 (H, W). Higher pressure -> skin lifts -> nearer."""
        d = self.base.depth_mm - self.pulse.depth_lift_mm(self.frame_time(i))
        return self.sensor.depth(d, noise=noise)
