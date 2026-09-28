"""Pressure to blood volume under the skin: each vessel's pressure field becomes a change in its lumen area, and that
change is spread over the skin as a lift, by the elastic response of the tissue above it. A uniform skin pulse carries
the microvascular blood volume of the skin itself. Nothing is coloured yet; the optical stage reads these fields.

The lift is that of a line source of area change dA at depth h_c below a free surface of incompressible tissue,
u(s) = dA h_c / (pi (h_c^2 + s^2)) at lateral distance s, a Lorentzian of half-width h_c whose integral over the
surface is dA, so the blood displaced by the vessel is conserved and a deep vessel lifts a broad low strip while a
shallow one lifts a narrow high one. On the cylinder h_c^2 + s^2 is the squared distance from the skin point to the
vessel axis that the scene maps already carry. The lift is also the thickness of extra blood under the pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.abp import abp_shape
from ..traces.ecg import t_wave_end
from ..traces.priors import Normal, Range
from .propagation import Propagation, PulseField
from .scene import Scene, SceneMaps

KPA_PER_MMHG = 0.133322


@dataclass(frozen=True)
class DistensionConfig:
    carotid_compliance_mm2_per_kpa: Normal     # lumen area change per unit pressure, common carotid
    jugular_area_strain_per_mmhg: Normal       # fractional lumen area change per mmHg, internal jugular
    skin_transit_s: Range                      # carotid foot at the neck to the foot of the skin pulse


@dataclass(frozen=True)
class Distension:
    carotid_compliance_mm2_per_kpa: float
    jugular_area_strain_per_mmhg: float
    skin_transit_s: float


def draw_distension(cfg: DistensionConfig, rng: np.random.Generator) -> Distension:
    return Distension(cfg.carotid_compliance_mm2_per_kpa.draw(rng), cfg.jugular_area_strain_per_mmhg.draw(rng),
                      cfg.skin_transit_s.draw(rng))


def skin_pulse(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float, duration_s: float,
               pre_ejection_s: float, prop: Propagation, dist: Distension, am_fraction: float,
               wander: float) -> np.ndarray:
    """Dimensionless skin blood-volume pulse on the grid `t`, uniform over the neck: the finger PPG shape with its foot
    `dist.skin_transit_s` after the carotid foot, rescaled over 0 <= t < duration_s to peak-to-trough 1 about a zero
    mean, then modulated by respiration as the PPG is, (1 - am_fraction x) - wander x, without noise. Its scale in
    blood is set by the optical stage."""
    r = np.asarray(r_times, dtype=np.float64)
    ejection = t_wave_end(r, heart_rate_bpm) - r[:-1] - pre_ejection_s
    feet = r[:-1] + pre_ejection_s + prop.aortic_to_carotid_s + dist.skin_transit_s
    s = abp_shape(t, feet, ejection, "finger")
    win = (t >= 0.0) & (t < duration_s)
    s = (s - s[win].min()) / (s[win].max() - s[win].min())
    base = s - s[win].mean()
    resp = np.asarray(resp, dtype=np.float64)
    return base * (1.0 - am_fraction * resp) - wander * resp


def lift_kernel(axis_depth_mm: float, dist_mm: np.ndarray) -> np.ndarray:
    """Surface lift per unit area change (1/mm) at distance `dist_mm` from a line source `axis_depth_mm` below the
    skin: h / (pi d^2). Directly over the source d = h and the lift is 1 / (pi h); at a lateral distance h it is half
    that; the integral over the surface is 1."""
    return axis_depth_mm / (np.pi * np.asarray(dist_mm, dtype=np.float64) ** 2)


class VolumeField:
    """Blood volume changes at every pixel at any time: the lift of the skin over each vessel (mm), the depth of the
    lifted skin, and the uniform skin pulse. Area changes are about the mean pressure of each trace over the nominal
    duration, so the drawn vessel diameters are the sizes at mean pressure."""

    def __init__(self, pulse: PulseField, scene: Scene, maps: SceneMaps, dist: Distension, duration_s: float,
                 skin: np.ndarray):
        self.pulse, self.maps = pulse, maps
        self.skin = np.asarray(skin, dtype=np.float64)
        win = (pulse.t >= 0.0) & (pulse.t < duration_s)
        self.artery_mean_mmhg = float(pulse.carotid[win].mean())
        self.vein_mean_mmhg = float(pulse.cvp[win].mean())
        self.artery_compliance_mm2_per_mmhg = dist.carotid_compliance_mm2_per_kpa * KPA_PER_MMHG
        vein_area = np.pi * (0.5 * scene.vein_diameter_mm) ** 2
        self.vein_compliance_mm2_per_mmhg = dist.jugular_area_strain_per_mmhg * vein_area
        r = scene.neck_radius_mm
        self.artery_axis_depth_mm = r - scene.artery_axis_radius_mm
        self.vein_axis_depth_mm = r - scene.vein_axis_radius_mm
        self.artery_kernel = np.where(maps.on_neck, lift_kernel(self.artery_axis_depth_mm, maps.artery_dist_mm), 0.0)
        self.vein_kernel = np.where(maps.on_neck, lift_kernel(self.vein_axis_depth_mm, maps.vein_dist_mm), 0.0)
        self.normal_z = np.where(maps.on_neck, np.cos(maps.phi), 0.0)   # outward normal towards the camera
        # d(kernel)/ds along the arc s = r phi, for the tilt of the surface: -h 2 rho sin(phi - phi_v) / (pi d^4)
        self.artery_slope = np.where(maps.on_neck, -self.artery_axis_depth_mm * 2 * scene.artery_axis_radius_mm
                                     * np.sin(maps.phi - scene.artery_phi) / (np.pi * maps.artery_dist_mm ** 4), 0.0)
        self.vein_slope = np.where(maps.on_neck, -self.vein_axis_depth_mm * 2 * scene.vein_axis_radius_mm
                                   * np.sin(maps.phi - scene.vein_phi) / (np.pi * maps.vein_dist_mm ** 4), 0.0)

    def artery_area_mm2(self, time_s: float) -> np.ndarray:
        return self.artery_compliance_mm2_per_mmhg * (self.pulse.artery_mmhg(time_s) - self.artery_mean_mmhg)

    def vein_area_mm2(self, time_s: float) -> np.ndarray:
        return self.vein_compliance_mm2_per_mmhg * (self.pulse.vein_mmhg(time_s) - self.vein_mean_mmhg)

    def artery_lift_mm(self, time_s: float) -> np.ndarray:
        return self.artery_area_mm2(time_s) * self.artery_kernel

    def vein_lift_mm(self, time_s: float) -> np.ndarray:
        return self.vein_area_mm2(time_s) * self.vein_kernel

    def depth_mm(self, time_s: float) -> np.ndarray:
        """Depth of the lifted skin: the lift is along the outward normal, so the camera sees its component towards
        it, and a lift brings the skin nearer."""
        return self.maps.depth_mm - (self.artery_lift_mm(time_s) + self.vein_lift_mm(time_s)) * self.normal_z

    def tilt(self, time_s: float) -> np.ndarray:
        """Rotation of the outward normal in the cross-section plane (radians, positive towards larger phi) by the
        slope of the lift along the arc: the normal of a surface u(s) is turned by -du/ds."""
        return -(self.artery_area_mm2(time_s) * self.artery_slope + self.vein_area_mm2(time_s) * self.vein_slope)

    def skin_pulse(self, time_s: float) -> float:
        return float(np.interp(time_s, self.pulse.t, self.skin))


def quantise_depth(depth_mm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Depth in integer millimetres as uint16, by stochastic rounding: a value of 700.3 is 701 with probability 0.3 and
    700 otherwise, so a lift smaller than a millimetre still moves that fraction of the pixels."""
    d = np.asarray(depth_mm, dtype=np.float64)
    return np.floor(d + rng.random(d.shape)).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
