import json

import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.config import GeneratorConfig, apply_override
from synthetic_neck.generate import generate_dataset, generate_sample
from synthetic_neck.render.camera import to_gray
from synthetic_neck.video import read_mkv


def _small():
    return apply_override(GeneratorConfig(), "video.frame_size", "64")


@needs_ffmpeg
def test_generate_sample_end_to_end(tmp_path):
    meta = generate_sample(_small(), seed=7, out_dir=tmp_path / "1", preset="lesson")
    d = tmp_path / "1"
    assert {p.name for p in d.iterdir()} == {"trace.csv", "gray_video.mkv", "rgbid_video.mkv",
                                             "vessel_ids.npy", "metadata.json"}
    assert meta["n_frames"] == 300 and meta["preset"] == "lesson" and meta["seed"] == 7
    assert meta["streams"] == ["rgb", "ir", "depth"]
    rgb = read_mkv(d / "rgbid_video.mkv", "rgb", stream=0)
    gray = read_mkv(d / "gray_video.mkv", "gray")
    assert rgb.shape == (300, 64, 64, 3) and gray.shape == (300, 64, 64)
    np.testing.assert_array_equal(gray, np.stack([to_gray(f) for f in rgb]))
    ir = read_mkv(d / "rgbid_video.mkv", "gray", stream=1)
    depth = read_mkv(d / "rgbid_video.mkv", "gray16", stream=2)
    assert ir.shape == (300, 64, 64) and depth.dtype == np.uint16
    assert 400 < (depth * 0.02).mean() < 1000
    ids = np.load(d / "vessel_ids.npy")
    assert set(np.unique(ids)) <= {0, 1, 2}
    saved = json.loads((d / "metadata.json").read_text())
    assert saved["config"]["video"]["frame_size"] == 64
    assert saved["params"]["trace"]["heart_rate_bpm"] == meta["params"]["trace"]["heart_rate_bpm"]
    assert 0.5 < saved["derived"]["pixel_scale_mm"] < 1.2


@needs_ffmpeg
def test_sample_without_depth_ir_streams(tmp_path):
    cfg = apply_override(_small(), "streams.depth_ir_probability", "0")
    meta = generate_sample(cfg, seed=1, out_dir=tmp_path / "1")
    assert meta["streams"] == ["rgb"]
    import subprocess
    n = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(tmp_path / "1" / "rgbid_video.mkv")], capture_output=True, text=True).stdout.split()
    assert len(n) == 1


@needs_ffmpeg
def test_generate_dataset_writes_index_and_reports_failures(tmp_path, monkeypatch):
    import synthetic_neck.generate as gen
    real = gen.generate_sample

    def flaky(config, seed, out_dir, preset="custom"):
        if seed == 101:
            raise RuntimeError("boom")
        return real(config, seed, out_dir, preset)

    monkeypatch.setattr(gen, "generate_sample", flaky)
    results = gen.generate_dataset(_small(), tmp_path, n=2, start=1, base_seed=100, preset="lesson",
                                   overrides=["video.frame_size=64"], jobs=1)
    assert [(i, e is None) for i, e in results] == [(1, False), (2, True)]
    assert not (tmp_path / "1").exists() and (tmp_path / "2" / "metadata.json").exists()
    idx = json.loads((tmp_path / "dataset.json").read_text())
    assert idx["preset"] == "lesson" and idx["overrides"] == ["video.frame_size=64"] and idx["base_seed"] == 100
    assert idx["failed"] == [{"index": 1, "seed": 101}]


@needs_ffmpeg
def test_generate_dataset_parallel_jobs(tmp_path):
    results = generate_dataset(_small(), tmp_path, n=2, start=1, base_seed=200, preset="lesson",
                               overrides=[], jobs=2)
    assert all(e is None for _, e in results)
    assert (tmp_path / "1" / "metadata.json").exists() and (tmp_path / "2" / "metadata.json").exists()
    meta1 = json.loads((tmp_path / "1" / "metadata.json").read_text())
    meta2 = json.loads((tmp_path / "2" / "metadata.json").read_text())
    assert meta1["seed"] == 201 and meta2["seed"] == 202
    idx = json.loads((tmp_path / "dataset.json").read_text())
    assert idx["failed"] == []


@needs_ffmpeg
def test_visibility_guarantee_redraws_until_pulse_is_testable(tmp_path, monkeypatch):
    import synthetic_neck.generate as gen
    calls = []
    real = gen.cardiac_snr

    def fake(series, fps, hr_hz):
        calls.append(1)
        return 1.0 if len(calls) <= 2 else real(series, fps, hr_hz)   # first attempt: artery and vein fail

    monkeypatch.setattr(gen, "cardiac_snr", fake)
    meta = gen.generate_sample(_small(), seed=7, out_dir=tmp_path / "1")
    assert meta["visibility"]["attempts"] == 2
    assert meta["visibility"]["artery_snr"] >= 10 and meta["visibility"]["vein_snr"] >= 10
    first = gen.sample(_small(), gen._attempt_rng(7, 0))
    assert meta["params"]["trace"]["heart_rate_bpm"] != first.trace.heart_rate_bpm


@needs_ffmpeg
def test_visibility_guarantee_gives_up_after_max_attempts(tmp_path, monkeypatch):
    import synthetic_neck.generate as gen
    monkeypatch.setattr(gen, "cardiac_snr", lambda series, fps, hr_hz: 0.0)
    monkeypatch.setattr(gen, "MAX_VISIBILITY_ATTEMPTS", 2)
    with pytest.raises(RuntimeError, match="no draw reached"):
        gen.generate_sample(_small(), seed=7, out_dir=tmp_path / "1")
