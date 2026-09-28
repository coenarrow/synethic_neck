"""A frame at a time: the static appearance, re-lit with the tilt of the lifted skin and coloured by the skin pulse,
with the depth of the lifted skin. Continuous values on the render grid; the sensor stage samples and quantises."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import Appearance, BaseFrame, shading
from .distension import VolumeField
from .optics import Optics, pulse_factor
from .scene import SceneMaps


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray                 # (n, n, 3) float, 8-bit levels
    ir: np.ndarray                  # (n, n) float, 16-bit levels
    depth_mm: np.ndarray            # (n, n) float


def render_frame(time_s: float, maps: SceneMaps, app: Appearance, base: BaseFrame, vol: VolumeField,
                 opt: Optics) -> Frame:
    """The frame at `time_s`: on the neck, the swatch times the shading of the tilted surface times the pulse factor
    of each band, all under the base texture, the infrared lit from the camera; beside it the base frame unchanged."""
    tilt = vol.tilt(time_s)
    sh, sh_ir = shading(maps, app, tilt), shading(maps, app, tilt, light_angle_deg=0.0)
    fac = pulse_factor(opt, vol.skin_pulse(time_s))
    colour = np.stack([fac["r"], fac["g"], fac["b"]]) * np.asarray(app.skin_rgb)
    skin = colour[None, None, :] * sh[..., None] * base.texture[..., None]
    rgb = np.where(maps.on_neck[..., None], skin, base.rgb)
    ir = np.where(maps.on_neck, app.ir_level * fac["ir"] * sh_ir * base.texture, base.ir)
    return Frame(rgb, ir, vol.depth_mm(time_s))
