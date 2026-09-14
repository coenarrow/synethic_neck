import json
from dataclasses import replace

import numpy as np
import pytest

from synthetic_neck.config import (
    Choice, ConfigError, GeneratorConfig, IntRange, Range, apply_override, config_from_dict,
    config_to_dict, params_to_dict, sample, validate, vein_visible_fraction,
)


def test_range_draws_within_bounds_and_fixed_when_degenerate():
    rng = np.random.default_rng(0)
    r = Range(2.0, 3.0)
    xs = [r.draw(rng) for _ in range(100)]
    assert all(2.0 <= x <= 3.0 for x in xs) and min(xs) < 2.2 and max(xs) > 2.8
    assert Range(5.0, 5.0).draw(rng) == 5.0
    assert IntRange(1, 3).draw(rng) in (1, 2, 3)
    assert Choice((0, 45, 90), (0.0, 0.0, 1.0)).draw(rng) == 90
    with pytest.raises(ConfigError):
        Range(3.0, 2.0)


def test_default_config_validates_and_samples():
    cfg = GeneratorConfig()
    validate(cfg)
    p = sample(cfg, np.random.default_rng(1))
    assert p.video.frame_size == 300 and p.trace.duration_s == 10.0
    assert 55 <= p.trace.heart_rate_bpm <= 95
    assert p.trace.systolic_mmhg > p.trace.diastolic_mmhg
    assert p.geometry.artery_side in (-1, 1)
    assert 0.4 <= p.geometry.centre_frac_xy[0] <= 0.6
    assert p.streams.has_depth_ir is True
    assert len(p.appearance.skin_rgb) == 3


def test_sampling_is_deterministic():
    cfg = GeneratorConfig()
    a = params_to_dict(sample(cfg, np.random.default_rng(9)))
    b = params_to_dict(sample(cfg, np.random.default_rng(9)))
    assert a == b


def test_override_parsing():
    cfg = GeneratorConfig()
    cfg = apply_override(cfg, "pulse.amplitude_levels", "0.5,2")
    assert cfg.pulse.amplitude_levels == Range(0.5, 2.0)
    cfg = apply_override(cfg, "pulse.amplitude_levels", "4")
    assert cfg.pulse.amplitude_levels == Range(4.0, 4.0)
    cfg = apply_override(cfg, "video.frame_size", "64")
    assert cfg.video.frame_size == 64
    cfg = apply_override(cfg, "trace.posture_deg", "0|45")
    assert cfg.trace.posture_deg.values == (0.0, 45.0)
    cfg = apply_override(cfg, "streams.depth_ir_probability", "0.5")
    assert cfg.streams.depth_ir_probability == 0.5
    with pytest.raises(ConfigError, match="unknown"):
        apply_override(cfg, "pulse.nope", "1")
    with pytest.raises(ConfigError, match="unknown"):
        apply_override(cfg, "nope.x", "1")
    with pytest.raises(ConfigError):
        apply_override(cfg, "pulse.amplitude_levels", "abc")


def test_validation_rejects_bad_configs():
    cfg = apply_override(GeneratorConfig(), "geometry.vein_visible_fraction", "0.05,1")
    with pytest.raises(ConfigError, match="vein_visible_fraction"):
        validate(cfg)
    cfg = apply_override(GeneratorConfig(), "geometry.separation_mm", "400,400")
    with pytest.raises(ConfigError, match="fit"):
        validate(cfg)
    with pytest.raises(ConfigError, match="fps"):
        validate(apply_override(GeneratorConfig(), "video.fps", "10"))
    with pytest.raises(ConfigError, match="depth"):
        validate(apply_override(GeneratorConfig(), "camera.distance_mm", "500,1400"))


def test_json_round_trip():
    cfg = apply_override(GeneratorConfig(), "illumination.flicker_amp", "0.01,0.02")
    d = config_to_dict(cfg)
    assert json.loads(json.dumps(d)) == d
    assert config_from_dict(d) == cfg


def test_knotted_range_follows_quantiles():
    r = Range(0.0, 10.0, (0.0, 1.0, 2.0, 3.0, 10.0))
    rng = np.random.default_rng(0)
    draws = np.array([r.draw(rng) for _ in range(4000)])
    assert abs(np.median(draws) - 2.0) < 0.1
    share = float(np.mean(draws > 3.0))
    assert 0.18 <= share <= 0.26
    with pytest.raises(ConfigError):
        Range(0.0, 1.0, (0.0, 0.5, 0.4, 0.8, 1.0))


def test_knotted_range_json_round_trip():
    cfg = apply_override(GeneratorConfig(), "illumination.flicker_amp", "0.01,0.02")
    cfg = replace(cfg, sensor=replace(cfg.sensor, read_noise_sd=Range(1.0, 5.0, (1.0, 2.0, 3.0, 4.0, 5.0))))
    d = config_to_dict(cfg)
    assert json.loads(json.dumps(d)) == d
    assert config_from_dict(d) == cfg


def test_plain_range_stream_unchanged():
    assert Range(2.0, 3.0).draw(np.random.default_rng(5)) == float(np.random.default_rng(5).uniform(2.0, 3.0))


def test_vein_visible_fraction():
    b = Range(0.4, 1.0)
    assert vein_visible_fraction(0, 6.0, 60.0, b) == 1.0          # supine: whole column visible
    upright = vein_visible_fraction(90, 6.0, 60.0, b)
    mid = vein_visible_fraction(45, 6.0, 60.0, b)
    assert 0.4 <= upright <= mid <= 1.0
    assert vein_visible_fraction(90, 15.0, 60.0, b) > upright     # higher CVP -> taller column
