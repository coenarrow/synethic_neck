import numpy as np
import pytest

from synthetic_neck.config import IlluminationParams
from synthetic_neck.render.illumination import IlluminationStage


def _stage(**kw):
    shading = np.ones((50, 50))
    return IlluminationStage(IlluminationParams(**kw), 50, 30.0, 300, shading)


def test_identity_with_default_params():
    st = _stage()
    img = np.random.default_rng(0).random((50, 50, 3)) * 255
    np.testing.assert_array_equal(st(img, 0), img)
    np.testing.assert_array_equal(st(img, 299), img)


def test_ambient_gain_scales():
    st = _stage(ambient_gain=0.5)
    img = np.full((50, 50), 100.0)
    np.testing.assert_allclose(st(img, 7), 50.0)


def test_drift_is_slow_bounded_and_seeded():
    a = _stage(drift_sd=0.05, drift_tau_s=20.0, seed=1)
    b = _stage(drift_sd=0.05, drift_tau_s=20.0, seed=1)
    gains = np.array([a.gain(i) for i in range(300)])
    assert np.array_equal(gains, [b.gain(i) for i in range(300)])
    assert np.abs(gains - 1).max() < 0.3
    assert np.abs(np.diff(gains)).max() < 0.02      # no frame-to-frame jumps


def test_flicker_aliases_to_a_periodic_gain():
    st = _stage(flicker_amp=0.02, flicker_hz=100.0)
    gains = np.array([st.gain(i) for i in range(300)])
    # 100 Hz sampled at 30 fps aliases to a 3-frame period whose samples sit at sin(0), sin(±2π/3).
    assert np.all(np.abs(gains - 1.0) <= 0.02 + 1e-12)
    np.testing.assert_allclose(gains[:-3], gains[3:])
    assert np.ptp(gains) > 0.03


def test_specular_blob_sits_on_shading_peak():
    shading = np.ones((50, 50)) * 0.5
    shading[:, 30] = 1.0            # cylinder axis at column 30
    st = IlluminationStage(IlluminationParams(specular_amp=20.0, specular_sigma_frac=0.1), 50, 30.0, 300, shading)
    assert st.specular.max() == pytest.approx(20.0)
    assert np.unravel_index(np.argmax(st.specular), st.specular.shape) == (25, 30)
    img = np.zeros((50, 50, 3))
    assert st(img, 0)[25, 30, 1] == pytest.approx(20.0)
