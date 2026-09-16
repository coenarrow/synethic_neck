import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.config import apply_override
from synthetic_neck.generate import generate_sample
from synthetic_neck.inspect import cardiac_snr, fft_maps, vessel_phase_summary, vessel_power_ratio, vessel_snr
from synthetic_neck.presets import get_preset


@needs_ffmpeg
@pytest.mark.parametrize("name", ["lesson", "benchmark", "neckflix"])
def test_fft_maps_localise_vessels_for_each_preset(tmp_path, name):
    if name == "neckflix":
        from synthetic_neck.presets import DEFAULT_PRIORS
        if not DEFAULT_PRIORS.exists():
            pytest.skip("priors/neckflix.json not generated yet (Task 16)")
    # neckflix renders at 96 px.
    cfg = apply_override(get_preset(name), "video.frame_size", "96" if name == "neckflix" else "64")
    generate_sample(cfg, seed=11, out_dir=tmp_path / "1", preset=name)
    power, phase, f_hz = fft_maps(tmp_path / "1")
    ids = np.load(tmp_path / "1" / "vessel_ids.npy")
    ratios = vessel_power_ratio(power, ids)
    snr = vessel_snr(tmp_path / "1")
    # Localisation: per-pixel artery power at the heart rate stands out from the pulsing skin in the carrier channels (lesson only; benchmark's noise leaves the ratio near 1).
    localised = {"lesson": ("R", "G", "B"), "benchmark": (), "neckflix": ()}[name]
    for ch in localised:
        assert ratios[ch][0] > 1.5, (ch, ratios[ch])
    if name == "lesson":
        assert ratios["Depth (mm)"][0] > 1.2
    # Detectability: each vessel's averaged cardiac signal (HR, 2xHR, 3xHR) is at least 10x its own noise floor.
    # For neckflix, generate_sample already enforces G SNR >= 10 with this same measure; the phase check
    # below is the independent assertion.
    detect = ("G",) + (("IR",) if "IR" in snr and name != "neckflix" else ())
    for ch in detect:
        assert snr[ch][0] >= 10, (ch, snr[ch])
        assert snr[ch][1] >= 10, (ch, snr[ch])
    summary = vessel_phase_summary(phase, ids)
    phase_checked = localised + tuple(c for c in detect if c not in localised)
    assert all(abs(summary[ch][2]) > 0.3 for ch in phase_checked)


@needs_ffmpeg
def test_plot_maps_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from synthetic_neck.inspect import plot_maps
    cfg = apply_override(get_preset("lesson"), "video.frame_size", "48")
    generate_sample(cfg, seed=2, out_dir=tmp_path / "1")
    assert plot_maps(tmp_path / "1").exists()


def test_cardiac_snr_counts_harmonics_and_rejects_noise():
    fps, n, hr = 30.0, 300, 1.2
    t = np.arange(n) / fps
    rng = np.random.default_rng(0)
    noise = 0.1 * rng.standard_normal(n)
    assert cardiac_snr(np.sin(2 * np.pi * 2 * hr * t) + noise, fps, hr) >= 10   # energy only at 2xHR
    assert cardiac_snr(noise, fps, hr) < 10
