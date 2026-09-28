"""Pulse propagation: the traces become a pressure at every neck pixel at every frame time.

The carotid pressure is the ABP Windkessel with the carotid row, its foot an aortic-to-carotid transit after the
valve opens, defined at the caudal end of the visible segment; along the segment each pixel reads it delayed by its
axial distance over the carotid pulse wave velocity. The jugular reads the central venous pressure delayed by the
measured atrium-to-neck lag plus the axial distance over the venous pulse wave velocity. Nothing is painted here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.abp import abp
from ..traces.priors import Normal, Range
from .scene import Scene, SceneMaps


@dataclass(frozen=True)
class PropagationConfig:
    aortic_to_carotid_s: Normal     # valve opening to the carotid foot at the neck
    carotid_pp_fraction: Range      # carotid pulse pressure as a fraction of the brachial
    carotid_pwv_m_s: Normal
    venous_pwv_m_s: Range
    atrium_to_neck_s: Normal        # lag of the jugular at the caudal end of the segment behind the CVP


@dataclass(frozen=True)
class Propagation:
    aortic_to_carotid_s: float
    carotid_pp_fraction: float
    carotid_pwv_m_s: float
    venous_pwv_m_s: float
    atrium_to_neck_s: float


def draw_propagation(cfg: PropagationConfig, rng: np.random.Generator) -> Propagation:
    return Propagation(cfg.aortic_to_carotid_s.draw(rng), cfg.carotid_pp_fraction.draw(rng),
                       cfg.carotid_pwv_m_s.draw(rng), cfg.venous_pwv_m_s.draw(rng), cfg.atrium_to_neck_s.draw(rng))


def carotid_pressure(t: np.ndarray, r_times: np.ndarray, resp: np.ndarray, heart_rate_bpm: float,
                     duration_s: float, map_mmhg: float, pulse_pressure_mmhg: float, pre_ejection_s: float,
                     prop: Propagation, resp_swing_mmhg: float) -> np.ndarray:
    """Carotid pressure (mmHg) at the caudal end of the segment on the grid `t`: the ABP model with the carotid
    shape, the foot `prop.aortic_to_carotid_s` after the valve opens, the pulse pressure scaled by
    `prop.carotid_pp_fraction`, the same respiratory swing as the arm and no measurement noise."""
    return abp(t, r_times, resp, heart_rate_bpm, duration_s, map_mmhg, pulse_pressure_mmhg, pre_ejection_s,
               prop.aortic_to_carotid_s, "carotid", prop.carotid_pp_fraction, resp_swing_mmhg, 0.0,
               np.random.default_rng(0))


class PulseField:
    """Pressure at every pixel of the frame at any time, for each vessel, from the padded 1000 Hz traces."""

    def __init__(self, t: np.ndarray, carotid_mmhg: np.ndarray, cvp_mmhg: np.ndarray, scene: Scene,
                 maps: SceneMaps, prop: Propagation):
        self.t = np.asarray(t, dtype=np.float64)
        self.carotid = np.asarray(carotid_mmhg, dtype=np.float64)
        self.cvp = np.asarray(cvp_mmhg, dtype=np.float64)
        axial_m = np.clip(maps.axial_mm, 0.0, scene.length_mm) * 1e-3
        self.artery_delay_s = axial_m / prop.carotid_pwv_m_s
        self.vein_delay_s = prop.atrium_to_neck_s + axial_m / prop.venous_pwv_m_s

    def artery_mmhg(self, time_s: float) -> np.ndarray:
        return np.interp(time_s - self.artery_delay_s, self.t, self.carotid)

    def vein_mmhg(self, time_s: float) -> np.ndarray:
        return np.interp(time_s - self.vein_delay_s, self.t, self.cvp)
