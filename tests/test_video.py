import numpy as np
from conftest import needs_ffmpeg

from synthetic_neck.video import MkvWriter, mux, read_mkv


@needs_ffmpeg
def test_mkv_round_trip_is_lossless(tmp_path):
    rng = np.random.default_rng(1)
    for fmt, shape, dtype, hi in (("gray", (5, 32, 48), np.uint8, 256), ("rgb", (5, 32, 48, 3), np.uint8, 256),
                                  ("gray16", (5, 32, 48), np.uint16, 65536)):
        frames = rng.integers(0, hi, size=shape, dtype=dtype)
        with MkvWriter(tmp_path / f"{fmt}.mkv", (32, 48), fmt) as w:
            for f in frames:
                w.write(f)
        np.testing.assert_array_equal(read_mkv(tmp_path / f"{fmt}.mkv", fmt), frames)
    mux([tmp_path / "rgb.mkv", tmp_path / "gray.mkv", tmp_path / "gray16.mkv"], tmp_path / "mux.mkv",
        titles=["rgb", "ir", "depth"])
    assert read_mkv(tmp_path / "mux.mkv", "gray16", stream=2).shape == (5, 32, 48)


@needs_ffmpeg
def test_writer_rejects_wrong_shape(tmp_path):
    import pytest
    with MkvWriter(tmp_path / "x.mkv", (4, 4), "gray") as w:
        with pytest.raises(ValueError):
            w.write(np.zeros((4, 5), np.uint8))
        w.write(np.zeros((4, 4), np.uint8))
