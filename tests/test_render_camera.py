import numpy as np
import pytest

from synthetic_neck.config import CameraParams, SensorParams
from synthetic_neck.render.camera import (
    SensorStage, area_resample_matrix, depth_to_uint16, gaussian_blur, to_gray,
)


def test_area_resample_preserves_mean_and_edges():
    m = area_resample_matrix(650, 300)
    np.testing.assert_allclose(m.sum(1), 1.0)
    img = np.random.default_rng(0).random((650, 650))
    out = m @ img @ m.T
    assert out.shape == (300, 300) and abs(out.mean() - img.mean()) < 1e-9


def test_gaussian_blur_identity_and_smoothing():
    img = np.zeros((41, 41))
    img[20, 20] = 1.0
    assert gaussian_blur(img, 0.0) is img
    b = gaussian_blur(img, 2.0)
    assert b.sum() == pytest.approx(1.0, abs=1e-6)
    assert b[20, 20] < 0.1 and b[20, 22] > 0.01
    rgb = np.stack([img] * 3, -1)
    assert gaussian_blur(rgb, 1.0).shape == rgb.shape


def test_sensor_identity_when_noise_and_blur_are_zero():
    cam = CameraParams(crop_px=64, output_px=32)
    st = SensorStage(SensorParams(read_noise_sd=0.0, shot_noise_gain=0.0, ir_read_noise_sd=0.0,
                                  blur_sigma_px=0.0, depth_noise_mm_at_1m=0.0), cam, 32)
    img = np.full((64, 64, 3), 100.4)
    out = st.rgb(img)
    assert out.dtype == np.uint8 and (out == 100).all()
    assert (st.ir(np.full((64, 64), 7.6)) == 8).all()
    np.testing.assert_allclose(st.depth(np.full((64, 64), 500.0)), 500.0)


def test_sensor_noise_matches_configured_sd():
    cam = CameraParams(crop_px=64, output_px=64)
    st = SensorStage(SensorParams(read_noise_sd=2.0, shot_noise_gain=0.0, seed=3), cam, 64)
    out = st.rgb(np.full((64, 64, 3), 128.0)).astype(float)
    assert out.std() == pytest.approx(2.0, abs=0.3)
    st = SensorStage(SensorParams(read_noise_sd=0.0, shot_noise_gain=0.05, seed=3), cam, 64)
    out = st.rgb(np.full((64, 64, 3), 200.0)).astype(float)
    assert out.std() == pytest.approx(np.sqrt(0.05 * 200), abs=0.4)
    st = SensorStage(SensorParams(depth_noise_mm_at_1m=1.6, seed=3), CameraParams(distance_mm=500, crop_px=64, output_px=64), 64)
    assert st.depth(np.full((64, 64), 500.0)).std() == pytest.approx(0.4, abs=0.05)


def test_depth_and_gray_conversions():
    assert depth_to_uint16(np.array([500.0])).tolist() == [25000]
    with pytest.raises(ValueError):
        depth_to_uint16(np.array([2000.0]))
    assert to_gray(np.array([[[255, 255, 255]]], np.uint8)).tolist() == [[255]]
