"""Vessel layout (pixels), per-pixel weights, visibility taper and propagation delays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CameraParams, GeometryParams

ID_BACKGROUND, ID_ARTERY, ID_VEIN = 0, 1, 2


@dataclass
class VesselGeometry:
    """Two parallel straight tubes in a square (frame_size) frame.

    The caudal end (s=0, nearest the heart) is `axis_start(which)`; waves travel
    towards `start + length * direction` (cranial). Widths are FWHM in pixels.
    """
    frame_size: int = 300
    length_px: float = 70.0
    separation_px: float = 25.0
    artery_width_px: float = 16.0
    vein_width_px: float = 18.0
    angle_deg: float = 80.0          # propagation direction, 0 = +x (right), 90 = up
    centre_xy: tuple[float, float] = (150.0, 150.0)
    artery_side: int = +1
    vein_visible_fraction: float = 1.0

    @property
    def direction(self) -> np.ndarray:
        a = np.deg2rad(self.angle_deg)
        return np.array([np.cos(a), -np.sin(a)])  # image y axis points down

    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])

    def axis_start(self, which: str) -> np.ndarray:
        side = self.artery_side if which == "artery" else -self.artery_side
        c = np.asarray(self.centre_xy) + side * 0.5 * self.separation_px * self.normal
        return c - 0.5 * self.length_px * self.direction


def vessel_geometry(gp: GeometryParams, camera: CameraParams) -> VesselGeometry:
    """Convert mm anatomy to output pixels at the camera's pixel scale."""
    n, k = camera.output_px, camera.pixel_scale_mm
    return VesselGeometry(
        frame_size=n,
        length_px=gp.length_mm / k,
        separation_px=gp.separation_mm / k,
        artery_width_px=gp.artery_width_mm / k,
        vein_width_px=gp.vein_width_mm / k,
        angle_deg=gp.angle_deg,
        centre_xy=(gp.centre_frac_xy[0] * n, gp.centre_frac_xy[1] * n),
        artery_side=gp.artery_side,
        vein_visible_fraction=gp.vein_visible_fraction,
    )


def _fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / 2.3548


def tube_fields(g: VesselGeometry, which: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight, s) maps of shape (H, W): Gaussian cross-section with soft
    end taper, and distance along the axis from the caudal end clipped to [0, L]."""
    n = g.frame_size
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    start = g.axis_start(which)
    rel_x, rel_y = xs - start[0], ys - start[1]
    d = g.direction
    s_raw = rel_x * d[0] + rel_y * d[1]
    perp = rel_x * g.normal[0] + rel_y * g.normal[1]
    width = g.artery_width_px if which == "artery" else g.vein_width_px
    sigma = _fwhm_to_sigma(width)
    cross = np.exp(-0.5 * (perp / sigma) ** 2)
    overshoot = np.maximum(0.0, np.maximum(-s_raw, s_raw - g.length_px))
    taper = np.exp(-0.5 * (overshoot / sigma) ** 2)
    s = np.clip(s_raw, 0.0, g.length_px)
    return cross * taper, s


def visibility_taper(s: np.ndarray, length_px: float, fraction: float) -> np.ndarray:
    """1 over the caudal `fraction` of the tube, smooth Gaussian fall-off beyond it."""
    if fraction >= 1.0:
        return np.ones_like(s)
    edge = fraction * length_px
    width = max(0.05 * length_px, 1.0)
    return np.where(s <= edge, 1.0, np.exp(-0.5 * ((s - edge) / width) ** 2))


def id_map(g: VesselGeometry) -> np.ndarray:
    """Label map: 0 background, 1 artery, 2 vein (inside the FWHM). The vein
    label covers the whole tube even where the pulse is tapered off."""
    ids = np.zeros((g.frame_size, g.frame_size), dtype=np.uint8)
    w_v, _ = tube_fields(g, "vein")
    w_a, _ = tube_fields(g, "artery")
    ids[w_v >= 0.5] = ID_VEIN
    ids[w_a >= 0.5] = ID_ARTERY
    return ids


def arterial_pwv_m_s(mean_pressure_mmhg: float) -> float:
    """Carotid PWV vs MAP: ~7.5 m/s at 90 mmHg, +0.04 m/s per mmHg."""
    return 4.0 + 0.04 * mean_pressure_mmhg


def venous_pwv_m_s(mean_pressure_mmhg: float) -> float:
    """Jugular PWV vs mean CVP: slow (~1-3 m/s), faster as the vein distends."""
    return 0.8 + 0.15 * mean_pressure_mmhg


def delay_map(s: np.ndarray, pwv_m_s: float, heart_to_neck_m: float, pixel_scale_mm: float) -> np.ndarray:
    """Seconds between the trace (measured near the heart) and each pixel."""
    return (heart_to_neck_m + s * pixel_scale_mm * 1e-3) / pwv_m_s


def scaled(g: VesselGeometry, k: float) -> VesselGeometry:
    """The same layout expressed in a frame k times larger."""
    return VesselGeometry(
        frame_size=int(round(g.frame_size * k)),
        length_px=g.length_px * k,
        separation_px=g.separation_px * k,
        artery_width_px=g.artery_width_px * k,
        vein_width_px=g.vein_width_px * k,
        angle_deg=g.angle_deg,
        centre_xy=(g.centre_xy[0] * k, g.centre_xy[1] * k),
        artery_side=g.artery_side,
        vein_visible_fraction=g.vein_visible_fraction,
    )
