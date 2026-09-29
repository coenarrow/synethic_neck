"""Frames in batches: the static appearance, re-lit with the tilt of the lifted skin and coloured by the skin pulse,
with the depth of the lifted skin, for every frame time in a batch at once. Continuous float32 values on the render
grid, on the backend's device; the sensor stage samples and quantises.

Per pixel and frame the render is: read each vessel's pressure at the frame time less the pixel's delay, turn it into
a lumen area change, and then the lift, the tilt of the normal and the depth are that area times the per-pixel fields
of the distension stage. Everything that does not depend on time is moved to the device once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.grid import TRACE_HZ
from .appearance import Appearance, BaseFrame
from .backend import Backend
from .distension import VolumeField
from .optics import Optics, pulse_factor
from .scene import SceneMaps


@dataclass(frozen=True)
class Frames:
    """A batch of rendered frames on the device: 8-bit levels of colour, 16-bit levels of infrared, mm of depth."""
    rgb: object                     # (B, 3, n, n) float32
    ir: object                      # (B, n, n)
    depth_mm: object                # (B, n, n)


class Renderer:
    def __init__(self, maps: SceneMaps, app: Appearance, base: BaseFrame, vol: VolumeField, opt: Optics,
                 backend: Backend):
        a = backend.array
        self.backend, self.opt = backend, opt
        self.n = maps.depth_mm.shape[0]
        pulse = vol.pulse
        self.trace_t, self.skin = pulse.t, vol.skin                         # numpy: the skin pulse is one value a frame
        self.carotid, self.cvp = a(pulse.carotid), a(pulse.cvp)
        self.t0 = float(pulse.t[0])
        # Sample positions in the padded 1 kHz traces are (t - t0) f - delay f: the per-pixel part is fixed.
        self.artery_delay = a(pulse.artery_delay_s * TRACE_HZ)
        self.vein_delay = a(pulse.vein_delay_s * TRACE_HZ)
        self.artery_mean, self.vein_mean = vol.artery_mean_mmhg, vol.vein_mean_mmhg
        self.artery_compliance, self.vein_compliance = vol.artery_compliance_mm2_per_mmhg, vol.vein_compliance_mm2_per_mmhg
        self.artery_slope, self.vein_slope = a(vol.artery_slope), a(vol.vein_slope)
        # The camera sees the lift's component along the ray, so fold the normal into the depth kernels.
        self.artery_depth_kernel = a(vol.artery_kernel * vol.normal_z)
        self.vein_depth_kernel = a(vol.vein_kernel * vol.normal_z)
        self.depth0 = a(maps.depth_mm)
        self.on_neck = a(maps.on_neck)
        self.phi_lit = a(maps.phi - np.deg2rad(app.light_angle_deg))     # the drawn light, for the colour
        self.phi_ir = a(maps.phi)                                        # the depth camera's own illuminator
        self.ambient, self.ir_level = app.ambient, app.ir_level
        self.skin_rgb = np.asarray(app.skin_rgb)
        self.texture = a(base.texture)
        self.base_rgb = a(np.moveaxis(base.rgb, -1, 0))                  # (3, n, n)
        self.base_ir = a(base.ir)

    def frames(self, times_s: np.ndarray) -> Frames:
        """The frames at `times_s` (B,), rendered together."""
        xp, a = self.backend.xp, self.backend.array
        times = np.asarray(times_s, dtype=np.float64)
        pos = a((times - self.t0) * TRACE_HZ)[:, None, None]
        area_a = self.artery_compliance * (self.backend.lookup(self.carotid, pos - self.artery_delay) - self.artery_mean)
        area_v = self.vein_compliance * (self.backend.lookup(self.cvp, pos - self.vein_delay) - self.vein_mean)
        tilt = -(area_a * self.artery_slope + area_v * self.vein_slope)
        depth = self.depth0 - (area_a * self.artery_depth_kernel + area_v * self.vein_depth_kernel)
        del area_a, area_v
        amb = self.ambient
        sh = (amb + (1.0 - amb) * xp.clip(xp.cos(self.phi_lit + tilt), 0.0, None)) * self.texture
        sh_ir = (amb + (1.0 - amb) * xp.clip(xp.cos(self.phi_ir + tilt), 0.0, None)) * self.texture
        del tilt
        fac = pulse_factor(self.opt, np.interp(times, self.trace_t, self.skin))    # (B,) per band, numpy
        colour = a(np.stack([fac["r"], fac["g"], fac["b"]], axis=1) * self.skin_rgb)    # (B, 3)
        rgb = xp.where(self.on_neck, colour[:, :, None, None] * sh[:, None], self.base_rgb)
        ir = xp.where(self.on_neck, a(self.ir_level * fac["ir"])[:, None, None] * sh_ir, self.base_ir)
        return Frames(rgb, ir, depth)
