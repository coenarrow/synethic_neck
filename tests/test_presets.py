import numpy as np
import pytest

from synthetic_neck.config import sample, validate
from synthetic_neck.presets import MONK_SKIN_RGB, PRESETS, benchmark, get_preset, lesson, neckflix


def test_preset_names():
    assert set(PRESETS) == {"lesson", "benchmark", "neckflix"}
    with pytest.raises(KeyError):
        get_preset("nope")


@pytest.mark.parametrize("name", ["lesson", "benchmark"])
def test_presets_validate_and_sample_50_seeds(name):
    cfg = get_preset(name)
    validate(cfg)
    for seed in range(50):
        p = sample(cfg, np.random.default_rng(seed))
        assert p.geometry.vein_visible_fraction >= cfg.geometry.vein_visible_fraction.lo
        assert p.pulse.amplitude_levels > 0


def test_amplitude_targets():
    assert lesson().pulse.amplitude_levels.lo == 7 and lesson().pulse.amplitude_levels.hi == 12
    assert benchmark().pulse.amplitude_levels.lo == 2 and benchmark().pulse.amplitude_levels.hi == 5
    assert lesson().geometry.vein_visible_fraction.lo == 1.0
    assert benchmark().geometry.vein_visible_fraction.lo == 0.6


def test_lesson_has_no_illumination_effects():
    il = lesson().illumination
    assert il.drift_sd.hi == 0 and il.flicker_amp.hi == 0 and il.specular_amp.hi == 0


def test_neckflix_without_priors_points_at_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        neckflix(tmp_path / "missing.json")


def test_monk_table_is_monotonic_in_brightness():
    lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in MONK_SKIN_RGB]
    assert len(MONK_SKIN_RGB) == 10 and all(a > b for a, b in zip(lum, lum[1:]))
