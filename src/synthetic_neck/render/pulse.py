"""Pulse stage: pressure -> additive pixel modulation with propagation delay."""
from __future__ import annotations

import numpy as np

from ..config import GeometryParams, PulseParams
from ..geometry import (VesselGeometry, arterial_pwv_m_s, delay_map, tube_fields, venous_pwv_m_s,
                        visibility_taper)


def normalise(x: np.ndarray) -> np.ndarray:
    """Map a trace to [-0.5, 0.5] using its 1st/99th percentiles (robust to noise)."""
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo), 0, 1) - 0.5


class PulseStage:
    def __init__(self, pulse: PulseParams, gp: GeometryParams, geometry: VesselGeometry,
                 pixel_scale_mm: float, trace: np.ndarray):
        self.p = pulse
        self.t = trace[:, 0]
        self.abp_n = normalise(trace[:, 1])
        self.cvp_n = normalise(trace[:, 2])
        self.map_art = float(trace[:, 1].mean())
        self.mean_cvp = float(trace[:, 2].mean())
        self.pwv_art = arterial_pwv_m_s(self.map_art)
        self.pwv_vein = venous_pwv_m_s(self.mean_cvp)

        self.w_art, s_art = tube_fields(geometry, "artery")
        w_vein, s_vein = tube_fields(geometry, "vein")
        self.w_vein = w_vein * visibility_taper(s_vein, geometry.length_px, geometry.vein_visible_fraction)
        self.delay_art = delay_map(s_art, self.pwv_art, gp.heart_to_neck_artery_m, pixel_scale_mm)
        self.delay_vein = delay_map(s_vein, self.pwv_vein, gp.heart_to_neck_vein_m, pixel_scale_mm)
        self._gain = np.asarray(pulse.channel_gain, dtype=np.float64)

    def pulse_maps(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised pressure at every pixel of each vessel at `time_s`."""
        p_art = np.interp(time_s - self.delay_art, self.t, self.abp_n)
        p_vein = np.interp(time_s - self.delay_vein, self.t, self.cvp_n)
        return p_art, p_vein

    def _green_mod(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        # Higher pressure -> more blood volume -> more absorption -> darker.
        return -(self.p.amplitude_levels * self.w_art * p_art
                 + self.p.vein_amplitude_levels * self.w_vein * p_vein)

    def rgb_mod(self, time_s: float) -> np.ndarray:
        return self._green_mod(time_s)[..., None] * self._gain[None, None, :]

    def ir_mod(self, time_s: float) -> np.ndarray:
        return self.p.ir_gain * self._green_mod(time_s)

    def depth_lift_mm(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        return self.p.artery_lift_mm * self.w_art * p_art + self.p.vein_lift_mm * self.w_vein * p_vein
