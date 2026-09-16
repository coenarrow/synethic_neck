"""Pulse stage: physiology -> additive pixel modulation.

The two vessel pressures propagate along their vessels with per-pixel delays. Two further terms are spatially
uniform: the skin-wide PPG (the stored finger PPG read at neck timing) darkens every pixel of the cylinder, and
breathing brightens the whole scene and lifts the whole frame. Design:
docs/superpowers/specs/2026-09-16-frame-physiology-design.md.
"""
from __future__ import annotations

import numpy as np

from ..config import GeometryParams, PulseParams, TraceParams
from ..geometry import (VesselGeometry, arterial_pwv_m_s, delay_map, tube_fields, venous_pwv_m_s,
                        visibility_taper)


def normalise(x: np.ndarray) -> np.ndarray:
    """Map a trace to [-0.5, 0.5] using its 1st/99th percentiles (robust to noise)."""
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo), 0, 1) - 0.5


class PulseStage:
    def __init__(self, pulse: PulseParams, gp: GeometryParams, geometry: VesselGeometry,
                 pixel_scale_mm: float, trace: np.ndarray, timing: TraceParams):
        self.p = pulse
        self.t = trace[:, 0]
        self.abp_n = normalise(trace[:, 1])
        self.cvp_n = normalise(trace[:, 2])
        self.ppg_n = normalise(trace[:, 4])
        self.rr = trace[:, 5]
        # The stored ABP is measured at an arm site `site_delay_s` after the aortic root; the carotid pulse is
        # the same waveform that much earlier, before its own root -> neck propagation (delay_art).
        self.site_delay_s = float(timing.abp_site_delay_s)
        self.map_art = float(trace[:, 1].mean())
        self.mean_cvp = float(trace[:, 2].mean())
        self.pwv_art = arterial_pwv_m_s(self.map_art)
        self.pwv_vein = venous_pwv_m_s(self.mean_cvp)

        self.w_art, s_art = tube_fields(geometry, "artery")
        w_vein, s_vein = tube_fields(geometry, "vein")
        self.w_vein = w_vein * visibility_taper(s_vein, geometry.length_px, geometry.vein_visible_fraction)
        self.delay_art = delay_map(s_art, self.pwv_art, gp.heart_to_neck_artery_m, pixel_scale_mm)
        self.delay_vein = delay_map(s_vein, self.pwv_vein, gp.heart_to_neck_vein_m, pixel_scale_mm)
        # The stored PPG is a finger pulse. The neck skin fills after the carotid's entry delay plus a capillary
        # transit, earlier than the finger, so its drive is the stored PPG read `skin_lead_s` ahead of frame time.
        self.entry_delay_art_s = float(gp.heart_to_neck_artery_m / self.pwv_art)
        self.r_to_skin_foot_s = float(timing.pep_s + self.entry_delay_art_s + timing.skin_transit_s)
        self.skin_lead_s = float(timing.r_to_ppg_foot_s - self.r_to_skin_foot_s)
        self._gain = np.asarray(pulse.channel_gain, dtype=np.float64)

    def pulse_maps(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Normalised pressure at every pixel of each vessel at `time_s`."""
        p_art = np.interp(time_s - self.delay_art + self.site_delay_s, self.t, self.abp_n)
        p_vein = np.interp(time_s - self.delay_vein, self.t, self.cvp_n)
        return p_art, p_vein

    def skin_drive(self, time_s: float) -> float:
        """Normalised PPG of the neck skin at `time_s`, uniform over the frame."""
        return float(np.interp(time_s + self.skin_lead_s, self.t, self.ppg_n))

    def resp_excursion(self, time_s: float) -> float:
        """Chest excursion in [-1, 1] at `time_s`; +1 at end-inspiration. No lag: the neck moves with the chest."""
        return float(2.0 * np.interp(time_s, self.t, self.rr) - 1.0)

    def resp_gain(self, time_s: float) -> float:
        """Whole-scene brightness factor: the neck tilts towards the light on inspiration."""
        return 1.0 + self.p.resp_gain_frac * self.resp_excursion(time_s)

    def _green_mod(self, time_s: float) -> np.ndarray:
        p_art, p_vein = self.pulse_maps(time_s)
        # More blood volume (higher pressure in a vessel, a fuller capillary bed) -> more absorption -> darker.
        return -(self.p.amplitude_levels * self.w_art * p_art
                 + self.p.vein_amplitude_levels * self.w_vein * p_vein
                 + self.p.skin_amplitude_levels * self.skin_drive(time_s))

    def rgb_mod(self, time_s: float) -> np.ndarray:
        return self._green_mod(time_s)[..., None] * self._gain[None, None, :]

    def ir_mod(self, time_s: float) -> np.ndarray:
        return self.p.ir_gain * self._green_mod(time_s)

    def depth_lift_mm(self, time_s: float) -> np.ndarray:
        """Lift towards the camera in mm: the vessel pulses, plus the whole neck rising on inspiration."""
        p_art, p_vein = self.pulse_maps(time_s)
        return (self.p.artery_lift_mm * self.w_art * p_art + self.p.vein_lift_mm * self.w_vein * p_vein
                + self.p.resp_lift_mm * self.resp_excursion(time_s))
