import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.config import apply_override
from synthetic_neck.generate import generate_sample
from synthetic_neck.inspect import fft_maps, vessel_phase_summary, vessel_power_ratio
from synthetic_neck.presets import get_preset


@needs_ffmpeg
@pytest.mark.parametrize("name", ["lesson", "benchmark", "neckflix"])
def test_fft_maps_localise_vessels_for_each_preset(tmp_path, name):
    if name == "neckflix":
        from synthetic_neck.presets import DEFAULT_PRIORS
        if not DEFAULT_PRIORS.exists():
            pytest.skip("priors/neckflix.json not generated yet (Task 16)")
    # neckflix amplitudes are near the noise floor, so give it more pixels per vessel.
    cfg = apply_override(get_preset(name), "video.frame_size", "96" if name == "neckflix" else "64")
    generate_sample(cfg, seed=11, out_dir=tmp_path / "1", preset=name)
    power, phase, f_hz = fft_maps(tmp_path / "1")
    ids = np.load(tmp_path / "1" / "vessel_ids.npy")
    ratios = vessel_power_ratio(power, ids)
    # Every preset must carry a detectable pulse in green (the designed carrier) and IR when recorded.
    # Red/blue/depth are only required to be clean in the lesson preset; elsewhere they are realistic.
    checked = ("R", "G", "B") if name == "lesson" else ("G",)
    checked += ("IR",) if "IR" in ratios else ()
    for ch in checked:
        assert ratios[ch][0] > 1.5, (ch, ratios[ch])      # artery vs background
        assert ratios[ch][1] > 1.2, (ch, ratios[ch])      # vein vs background
    if name == "lesson":
        assert ratios["Depth (mm)"][0] > 1.2
    summary = vessel_phase_summary(phase, ids)
    assert all(abs(summary[ch][2]) > 0.3 for ch in checked)


@needs_ffmpeg
def test_plot_maps_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from synthetic_neck.inspect import plot_maps
    cfg = apply_override(get_preset("lesson"), "video.frame_size", "48")
    generate_sample(cfg, seed=2, out_dir=tmp_path / "1")
    assert plot_maps(tmp_path / "1").exists()
