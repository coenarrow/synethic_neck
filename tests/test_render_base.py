import numpy as np
import pytest

from synthetic_neck.config import AppearanceParams, CameraParams
from synthetic_neck.geometry import VesselGeometry, tube_fields
from synthetic_neck.render.base import BaseScene


def _scene(**kw):
    g = VesselGeometry(frame_size=100, angle_deg=90, centre_xy=(50, 50), length_px=40)
    cam = CameraParams(distance_mm=500, crop_px=100, output_px=100)
    w_a, _ = tube_fields(g, "artery")
    w_v, _ = tube_fields(g, "vein")
    return BaseScene(AppearanceParams(texture_sd=0.0, vignette=0.0, static_vessel_contrast=0.0, **kw), g, cam, w_a, w_v)


def test_flat_scene_is_skin_colour():
    s = _scene(shading_strength=0.0, skin_rgb=(200.0, 150.0, 120.0))
    assert s.rgb.shape == (100, 100, 3)
    np.testing.assert_allclose(s.rgb[..., 0], 200.0)
    np.testing.assert_allclose(s.shading, 1.0)


def test_shading_peaks_on_cylinder_axis_and_falls_off():
    s = _scene(shading_strength=1.0, neck_radius_mm=30.0)
    # Vessel direction is vertical, so the cylinder axis is the vertical centre line: shading is
    # constant down a column and falls off left/right.
    assert s.shading[50, 50] == pytest.approx(1.0, abs=1e-6)
    assert s.shading[10, 50] == pytest.approx(s.shading[90, 50])
    assert s.shading[50, 5] < s.shading[50, 25] < s.shading[50, 50]
    # Depth is nearest on the axis and the shading peak coincides with the depth minimum.
    assert np.argmin(s.depth_mm[50]) == np.argmax(s.shading[50])
    assert s.depth_mm[50, 50] == pytest.approx(500 - 30)


def test_texture_is_reproducible_from_seed():
    g = VesselGeometry(frame_size=100)
    cam = CameraParams(crop_px=100, output_px=100)
    w_a, _ = tube_fields(g, "artery"); w_v, _ = tube_fields(g, "vein")
    s1 = BaseScene(AppearanceParams(seed=5, texture_sd=2.0), g, cam, w_a, w_v)
    s2 = BaseScene(AppearanceParams(seed=5, texture_sd=2.0), g, cam, w_a, w_v)
    np.testing.assert_array_equal(s1.rgb, s2.rgb)
    assert s1.rgb[..., 1].std() > 1.0
