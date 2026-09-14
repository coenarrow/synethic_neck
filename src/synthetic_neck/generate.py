"""Generate one sample or a whole dataset."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import GeneratorConfig, config_to_dict, params_to_dict, sample, validate
from .inspect import cardiac_snr
from .render.camera import DEPTH_UNITS_MM, depth_to_uint16, to_gray
from .render.renderer import Renderer
from .traces import generate_trace, read_trace_csv, write_trace_csv
from .video import MkvWriter, mux, require_ffmpeg

MIN_VISIBLE_SNR = 10.0
MAX_VISIBILITY_ATTEMPTS = 20


class SampleFailed(Exception):
    def __init__(self, index: int, seed: int, cause: Exception):
        super().__init__(f"sample {index} (seed {seed}) failed: {cause!r}")
        self.index, self.seed, self.cause = index, seed, cause


def _attempt_rng(seed: int, attempt: int) -> np.random.Generator:
    """Attempt 0 keeps the original stream for `seed`; later attempts use independent sub-streams."""
    return np.random.default_rng(seed if attempt == 0 else np.random.SeedSequence([seed, attempt]))


def _render(params, out_dir: Path) -> tuple[Renderer, list[str], float, float]:
    """Render every stream in one pass and return (renderer, streams, green artery SNR, green vein SNR)."""
    # 1. Ground-truth trace first; the video is rendered from the CSV on disk.
    write_trace_csv(generate_trace(params.trace), out_dir / "trace.csv")
    trace = read_trace_csv(out_dir / "trace.csv")

    # 2. Render every stream in one pass.
    r = Renderer(params, trace)
    n = params.video.frame_size
    size, fps = (n, n), params.video.fps
    has_di = params.streams.has_depth_ir
    rgb_tmp, ir_tmp, depth_tmp = (out_dir / f"_{s}.mkv" for s in ("rgb", "ir", "depth"))
    writers = [MkvWriter(rgb_tmp, size, "rgb", fps), MkvWriter(out_dir / "gray_video.mkv", size, "gray", fps)]
    if has_di:
        writers += [MkvWriter(ir_tmp, size, "gray", fps), MkvWriter(depth_tmp, size, "gray16", fps)]
    art_mask, vein_mask = r.ids == 1, r.ids == 2
    g_art, g_vein = np.empty(r.n_frames), np.empty(r.n_frames)
    try:
        for i in range(r.n_frames):
            rgb = r.frame(i)
            g = rgb[..., 1].astype(np.float64)
            g_art[i], g_vein[i] = g[art_mask].mean(), g[vein_mask].mean()
            writers[0].write(rgb)
            writers[1].write(to_gray(rgb))
            if has_di:
                writers[2].write(r.ir_frame(i))
                writers[3].write(depth_to_uint16(r.depth_frame(i)))
    finally:
        for w in writers:
            w.close()
    streams = ["rgb", "ir", "depth"] if has_di else ["rgb"]
    parts = [rgb_tmp, ir_tmp, depth_tmp] if has_di else [rgb_tmp]
    mux(parts, out_dir / "rgbid_video.mkv", titles=streams)
    for tmp in parts:
        tmp.unlink()
    np.save(out_dir / "vessel_ids.npy", r.ids)
    hr_hz = params.trace.heart_rate_bpm / 60.0
    return r, streams, cardiac_snr(g_art, fps, hr_hz), cardiac_snr(g_vein, fps, hr_hz)


def generate_sample(config: GeneratorConfig, seed: int, out_dir: Path, preset: str = "custom") -> dict:
    """Render one sample into `out_dir` and return its metadata dict. Guarantees a testable green-channel
    pulse: both vessels are checked and, if either is under MIN_VISIBLE_SNR, the sample is redrawn from an
    independent sub-stream of `seed`, up to MAX_VISIBILITY_ATTEMPTS times."""
    validate(config)
    require_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last = (0.0, 0.0)
    for attempt in range(MAX_VISIBILITY_ATTEMPTS):
        params = sample(config, _attempt_rng(seed, attempt))
        r, streams, artery_snr, vein_snr = _render(params, out_dir)
        last = (artery_snr, vein_snr)
        if artery_snr >= MIN_VISIBLE_SNR and vein_snr >= MIN_VISIBLE_SNR:
            break
    else:
        raise RuntimeError(f"seed {seed}: no draw reached green vessel SNR {MIN_VISIBLE_SNR} in "
                           f"{MAX_VISIBILITY_ATTEMPTS} attempts (last artery {last[0]:.1f}, vein {last[1]:.1f})")

    g, cam, pulse = r.geometry, params.camera, r.pulse
    meta = {
        "seed": seed,
        "preset": preset,
        "version": __version__,
        "frame_size": params.video.frame_size,
        "fps": params.video.fps,
        "n_frames": r.n_frames,
        "streams": streams,
        "config": config_to_dict(config),
        "params": params_to_dict(params),
        "geometry_px": {"length_px": g.length_px, "separation_px": g.separation_px,
                        "artery_width_px": g.artery_width_px, "vein_width_px": g.vein_width_px,
                        "centre_xy": list(g.centre_xy), "angle_deg": g.angle_deg},
        "derived": {
            "pixel_scale_mm": cam.pixel_scale_mm,
            "mean_abp_mmhg": pulse.map_art,
            "mean_cvp_mmhg": pulse.mean_cvp,
            "arterial_pwv_m_s": pulse.pwv_art,
            "venous_pwv_m_s": pulse.pwv_vein,
            "artery_delay_range_s": [float(pulse.delay_art.min()), float(pulse.delay_art.max())],
            "vein_delay_range_s": [float(pulse.delay_vein.min()), float(pulse.delay_vein.max())],
            "vein_visible_fraction": params.geometry.vein_visible_fraction,
        },
        "rgbid_streams": {str(i): s for i, s in enumerate(streams)},
        "depth_units_mm": DEPTH_UNITS_MM,
        "vessel_ids": {"file": "vessel_ids.npy", "0": "background", "1": "artery", "2": "vein"},
        "visibility": {"channel": "G", "min_snr": MIN_VISIBLE_SNR, "attempts": attempt + 1,
                       "artery_snr": artery_snr, "vein_snr": vein_snr},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _one(args) -> tuple[int, Exception | None]:
    config, index, seed, out_dir, preset = args
    try:
        generate_sample(config, seed, out_dir, preset)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller
        shutil.rmtree(out_dir, ignore_errors=True)
        return index, e


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:               # noqa: BLE001 - not a git checkout
        return None


def generate_dataset(config: GeneratorConfig, out_root: Path, n: int, start: int, base_seed: int,
                     preset: str, overrides: list[str], jobs: int = 1) -> list[tuple[int, Exception | None]]:
    """Generate samples start..start+n-1 (seed = base_seed + i) and write dataset.json."""
    validate(config)
    require_ffmpeg()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_args = [(config, i, base_seed + i, out_root / str(i), preset) for i in range(start, start + n)]
    if jobs <= 1:
        results = [_one(a) for a in jobs_args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_one, jobs_args))
    index = {
        "preset": preset,
        "overrides": list(overrides),
        "base_seed": base_seed,
        "start": start,
        "n": n,
        "version": __version__,
        "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config_to_dict(config),
        "failed": [{"index": i, "seed": base_seed + i} for i, e in results if e is not None],
    }
    (out_root / "dataset.json").write_text(json.dumps(index, indent=2))
    for i, e in results:
        if e is not None:
            print(f"sample {i} (seed {base_seed + i}) failed: {e!r}", file=sys.stderr)
    return results
