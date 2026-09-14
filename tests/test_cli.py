import json

import pytest
from conftest import needs_ffmpeg

from synthetic_neck.cli import build_config, main
from synthetic_neck.config import ConfigError, Range


def test_build_config_applies_preset_and_overrides():
    cfg = build_config("lesson", ["pulse.amplitude_levels=1,2", "video.frame_size=48"])
    assert cfg.pulse.amplitude_levels == Range(1, 2) and cfg.video.frame_size == 48
    with pytest.raises(ConfigError):
        build_config("lesson", ["pulse.nope=1"])
    with pytest.raises(ConfigError, match="key=value"):
        build_config("lesson", ["pulse.amplitude_levels"])


def test_bad_override_exits_nonzero_before_rendering(tmp_path, capsys):
    rc = main(["generate", "--preset", "lesson", "--n", "1", "--out", str(tmp_path), "--set", "pulse.nope=1"])
    assert rc == 2 and "unknown field" in capsys.readouterr().err
    assert not (tmp_path / "1").exists()


@needs_ffmpeg
def test_generate_command_writes_samples(tmp_path):
    rc = main(["generate", "--preset", "lesson", "--n", "2", "--seed", "5", "--out", str(tmp_path),
               "--set", "video.frame_size=48"])
    assert rc == 0
    assert (tmp_path / "1" / "metadata.json").exists() and (tmp_path / "2" / "metadata.json").exists()
    idx = json.loads((tmp_path / "dataset.json").read_text())
    assert idx["preset"] == "lesson" and idx["overrides"] == ["video.frame_size=48"]
    assert json.loads((tmp_path / "1" / "metadata.json").read_text())["seed"] == 6
