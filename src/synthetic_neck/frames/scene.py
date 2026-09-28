"""Scene geometry: a cylindrical neck with two straight vessels under its skin, and the per-pixel maps a frame is
built from. Nothing here moves; the traces enter at the pulse stage.

Camera coordinates: x right, y down, z forward (away from the camera). The cylinder axis lies in a plane
perpendicular to the optical axis, at the drawn distance plus the neck radius, so the ray through the frame
centre meets the skin at exactly the drawn distance. The vessels run parallel to the axis at an angular
position round the cylinder and a depth below the skin. The torso and neck are taken as aligned, so the axis lies
in the image at the subject's head-up posture angle, mirrored left or right at random.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.priors import Normal, Range
from .camera import Camera

ID_BACKGROUND, ID_ARTERY, ID_VEIN = 0, 1, 2


@dataclass(frozen=True)
class GeometryConfig:
    neck_circumference_cm: Normal
    length_mm: Range                # visible vessel segment
    separation_mm: Range            # arc between the two vessel axes along the skin
    artery_diameter_mm: Normal
    vein_diameter_mm: Normal
    artery_depth_mm: Normal         # skin to the anterior wall
    vein_depth_mm: Normal
    pair_offset_mm: Range           # arc from the frame centre to the midpoint between the vessels
    axial_offset_mm: Range          # along the axis from the frame centre to the midpoint of the segment
    posture_deg: Range              # head-up angle; the neck lies at this angle above the horizontal in the image, mirrored
                                    # left/right by a fair coin. Supine below 30, recumbent to 60, sitting above


@dataclass(frozen=True)
class Scene:
    neck_radius_mm: float
    length_mm: float
    angle_deg: float                # axis above the horizontal in the image: the posture, or its mirror
    axial_offset_mm: float
    artery_phi: float               # angle of each vessel axis round the cylinder, 0 facing the camera
    vein_phi: float
    artery_axis_radius_mm: float    # distance of each vessel axis from the cylinder axis
    vein_axis_radius_mm: float
    artery_diameter_mm: float
    vein_diameter_mm: float
    posture_deg: float

    @property
    def posture(self) -> str:
        return "supine" if self.posture_deg < 30 else "recumbent" if self.posture_deg < 60 else "sitting"

    @property
    def direction(self) -> np.ndarray:
        a = np.deg2rad(self.angle_deg)
        return np.array([np.cos(a), -np.sin(a), 0.0])      # image y points down

    @property
    def across(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0], 0.0])


def draw_scene(cfg: GeometryConfig, rng: np.random.Generator) -> Scene:
    r = 10.0 * cfg.neck_circumference_cm.draw(rng) / (2 * np.pi)
    length, sep, posture = cfg.length_mm.draw(rng), cfg.separation_mm.draw(rng), cfg.posture_deg.draw(rng)
    angle = posture if rng.random() < 0.5 else 180.0 - posture   # torso and neck aligned; either side of the neck
    d_art, d_vein = cfg.artery_diameter_mm.draw(rng), cfg.vein_diameter_mm.draw(rng)
    h_art, h_vein = cfg.artery_depth_mm.draw(rng), cfg.vein_depth_mm.draw(rng)
    pair, axial = cfg.pair_offset_mm.draw(rng), cfg.axial_offset_mm.draw(rng)
    side = 1.0 if rng.random() < 0.5 else -1.0              # which side of the pair the artery is on
    phi_mid = pair / r
    return Scene(r, length, angle, axial,
                 artery_phi=phi_mid + side * 0.5 * sep / r, vein_phi=phi_mid - side * 0.5 * sep / r,
                 artery_axis_radius_mm=r - h_art - 0.5 * d_art, vein_axis_radius_mm=r - h_vein - 0.5 * d_vein,
                 artery_diameter_mm=d_art, vein_diameter_mm=d_vein, posture_deg=posture)


@dataclass(frozen=True)
class SceneMaps:
    """Per-pixel geometry on an (n, n) grid. Distances in mm."""
    depth_mm: np.ndarray            # camera to skin (or to the background plane where the ray misses the neck)
    on_neck: np.ndarray             # bool
    phi: np.ndarray                 # angle round the cylinder of the skin point, 0 facing the camera
    axial_mm: np.ndarray            # along the axis from the caudal end of the vessel segment (unclipped)
    artery_dist_mm: np.ndarray      # skin point to the vessel axis, in the cross-section plane
    vein_dist_mm: np.ndarray
    labels: np.ndarray              # uint8: 0 background, 1 artery, 2 vein, over the projected lumen


def scene_maps(scene: Scene, camera: Camera, n: int) -> SceneMaps:
    v = camera.rays(n)                                       # (n, n, 3), unit z
    r = scene.neck_radius_mm
    d, across = scene.direction, scene.across
    c = np.array([0.0, 0.0, camera.distance_mm + r])         # a point on the cylinder axis, c . d = 0

    # Ray s*v meets the cylinder where |perp(s*v - c)|^2 = r^2, perp taken across the axis direction.
    v_perp = v - (v @ d)[..., None] * d[None, None, :]
    A = np.einsum("ijk,ijk->ij", v_perp, v_perp)
    B = v_perp @ c
    C = c @ c - r * r
    disc = B * B - A * C
    on_neck = disc >= 0
    s = np.where(on_neck, (B - np.sqrt(np.clip(disc, 0, None))) / A, camera.distance_mm + r)
    depth = s                                                # ray z is 1, so the parameter is the depth
    p = s[..., None] * v                                     # skin point in camera coordinates

    axial = p @ d                                            # along the axis; the centre ray has axial = 0
    q = p - c[None, None, :] - axial[..., None] * d[None, None, :]
    phi = np.arctan2(q @ across, -q[..., 2])                 # 0 facing the camera
    axial_from_caudal = axial - (scene.axial_offset_mm - 0.5 * scene.length_mm)

    def dist(phi_v, rho_v):
        return np.sqrt(r * r + rho_v * rho_v - 2 * r * rho_v * np.cos(phi - phi_v))

    art = dist(scene.artery_phi, scene.artery_axis_radius_mm)
    vein = dist(scene.vein_phi, scene.vein_axis_radius_mm)

    in_segment = (axial_from_caudal >= 0) & (axial_from_caudal <= scene.length_mm) & on_neck
    labels = np.zeros((n, n), dtype=np.uint8)
    labels[in_segment & (np.abs(r * (phi - scene.vein_phi)) <= 0.5 * scene.vein_diameter_mm)] = ID_VEIN
    labels[in_segment & (np.abs(r * (phi - scene.artery_phi)) <= 0.5 * scene.artery_diameter_mm)] = ID_ARTERY
    return SceneMaps(depth, on_neck, phi, axial_from_caudal, art, vein, labels)
