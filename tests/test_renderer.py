import dataclasses

import numpy as np

from synthetic_neck.config import GeneratorConfig, apply_override, sample
from synthetic_neck.geometry import tube_fields
from synthetic_neck.render.renderer import Renderer
from synthetic_neck.traces import generate_trace


def _params(seed=0, **overrides):
    cfg = GeneratorConfig()
    for k, v in overrides.items():
        cfg = apply_override(cfg, k, v)
    return sample(cfg, np.random.default_rng(seed))


def test_frame_shapes_and_types():
    p = _params(**{"video.frame_size": "64"})
    r = Renderer(p, generate_trace(p.trace))
    assert r.n_frames == 300
    assert r.frame(0).shape == (64, 64, 3) and r.frame(0).dtype == np.uint8
    assert r.ir_frame(0).shape == (64, 64) and r.ir_frame(0).dtype == np.uint8
    d = r.depth_frame(0)
    assert d.shape == (64, 64) and d.dtype == np.float64 and 400 < d.mean() < 1000
    assert set(np.unique(r.ids)) <= {0, 1, 2}


def test_frame_modulation_darkens_with_pressure():
    # Widest separation, narrowest vein: keeps the venous pulse from bleeding into the artery pixel.
    p = _params(**{"video.frame_size": "100", "trace.heart_rate_bpm": "60", "trace.hr_variability": "0",
                   "appearance.texture_sd": "0", "sensor.read_noise_sd": "0", "pulse.amplitude_levels": "10",
                   "geometry.separation_mm": "16", "geometry.vein_width_mm": "9"})
    tr = generate_trace(p.trace)
    r = Renderer(p, tr)
    w, _ = tube_fields(r.geometry, "artery")
    px = np.unravel_index(np.argmax(w), w.shape)
    green = np.array([r.frame(i, noise=False)[px][1] for i in range(r.n_frames)], dtype=float)
    delay = r.pulse.delay_art[int(px[0] * r.render_geometry.frame_size / 100), int(px[1] * r.render_geometry.frame_size / 100)]
    abp = np.array([np.interp(r.frame_time(i) - delay, tr[:, 0], tr[:, 1]) for i in range(r.n_frames)])
    assert np.corrcoef(green, abp)[0, 1] < -0.9
    assert 7 <= np.ptp(green) <= 13
    ir = np.array([r.ir_frame(i, noise=False)[px] for i in range(r.n_frames)], dtype=float)
    depth = np.array([r.depth_frame(i, noise=False)[px] for i in range(r.n_frames)])
    assert np.corrcoef(ir, abp)[0, 1] < -0.9 and np.ptp(ir) < np.ptp(green)
    assert np.corrcoef(depth, abp)[0, 1] < -0.9 and 0.15 < np.ptp(depth) < 0.4


def test_shading_and_depth_share_the_cylinder():
    p = _params(**{"video.frame_size": "64", "appearance.shading_strength": "1", "appearance.texture_sd": "0",
                   "appearance.vignette": "0", "sensor.read_noise_sd": "0", "pulse.amplitude_levels": "0",
                   "appearance.static_vessel_contrast": "0"})
    r = Renderer(p, generate_trace(p.trace))
    rgb = r.frame(0, noise=False).astype(float)[..., 0]
    depth = r.depth_frame(0, noise=False)
    # Brightest and nearest pixels lie on the same axis line: high correlation of -depth with brightness.
    assert np.corrcoef(rgb.ravel(), -depth.ravel())[0, 1] > 0.8


def test_identity_stages_reproduce_plain_render():
    p = _params(**{"video.frame_size": "48"})
    r = Renderer(p, generate_trace(p.trace))
    a = r.frame(5, noise=False)
    b = Renderer(p, generate_trace(p.trace)).frame(5, noise=False)
    np.testing.assert_array_equal(a, b)
