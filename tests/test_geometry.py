import numpy as np
import pytest

from synthetic_neck.config import CameraParams, GeneratorConfig, sample
from synthetic_neck.geometry import (
    VesselGeometry, delay_map, id_map, scaled, tube_fields, vessel_geometry, visibility_taper,
)


def test_camera_pixel_scale():
    cam = CameraParams(distance_mm=500)
    assert cam.native_scale_mm == pytest.approx(500 / 1920, rel=1e-6)
    assert cam.pixel_scale_mm == pytest.approx(500 / 1920 * 650 / 300, rel=1e-6)


def test_sampled_geometry_is_physical_and_fits_frame():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample(GeneratorConfig(), rng)
        scale = p.camera.pixel_scale_mm
        g = vessel_geometry(p.geometry, p.camera)
        assert g.frame_size == 300
        assert 40 <= g.length_px * scale <= 80
        assert 10 <= g.separation_px * scale <= 16
        assert 6 <= g.artery_width_px * scale <= 9 and 9 <= g.vein_width_px * scale <= 15
        assert 35 <= g.length_px <= 145
        ids = id_map(g)
        assert (ids == 1).any() and (ids == 2).any()
        assert ids[0].max() == 0 and ids[-1].max() == 0 and ids[:, 0].max() == 0 and ids[:, -1].max() == 0


def test_tube_profile_is_gaussian_with_fwhm():
    g = VesselGeometry(angle_deg=90, artery_width_px=20, centre_xy=(150, 150), separation_px=20, artery_side=1)
    w, s = tube_fields(g, "artery")
    axis_x = 160
    row = w[150]
    assert np.isclose(row[axis_x], 1.0)
    assert np.isclose(row[axis_x + 10], 0.5, atol=0.02)
    assert s[150, axis_x] == pytest.approx(35.0, abs=0.6)
    assert s[:, axis_x].min() == 0 and s[:, axis_x].max() == g.length_px


def test_visibility_taper_keeps_caudal_end_and_fades_cranial():
    s = np.linspace(0, 100, 101)
    t = visibility_taper(s, 100.0, 0.5)
    assert t[0] == pytest.approx(1.0) and t[40] == pytest.approx(1.0)
    assert t[60] < 0.6 and t[100] < 0.01
    np.testing.assert_allclose(visibility_taper(s, 100.0, 1.0), 1.0)


def test_scaled_geometry_keeps_layout():
    g = VesselGeometry(frame_size=300, length_px=70, centre_xy=(150, 140))
    k = scaled(g, 650 / 300)
    assert k.frame_size == 650 and k.length_px == pytest.approx(70 * 650 / 300)
    assert k.centre_xy[1] == pytest.approx(140 * 650 / 300)


def test_delay_grows_along_tube():
    s = np.array([0.0, 50.0, 100.0])
    d = delay_map(s, pwv_m_s=2.0, heart_to_neck_m=0.15, pixel_scale_mm=0.5)
    assert d[0] == pytest.approx(0.075) and d[2] == pytest.approx((0.15 + 0.05) / 2.0)
