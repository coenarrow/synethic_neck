"""Generate one sample or a whole dataset."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import GeneratorConfig, config_to_dict, params_to_dict, sample, validate
from .folder_store import FolderSink
from .zarr_store import ZarrSink, check_config
from .inspect import cardiac_snr
from .render.camera import DEPTH_UNITS_MM
from .render.renderer import Renderer
from .traces import generate_trace
from .video import require_ffmpeg

MIN_VISIBLE_SNR = 10.0
MAX_VISIBILITY_ATTEMPTS = 20


class SampleFailed(Exception):
    def __init__(self, index: int, seed: int, cause: Exception):
        super().__init__(f"sample {index} (seed {seed}) failed: {cause!r}")
        self.index, self.seed, self.cause = index, seed, cause


def _attempt_rng(seed: int, attempt: int) -> np.random.Generator:
    """Attempt 0 keeps the original stream for `seed`; later attempts use independent sub-streams."""
    return np.random.default_rng(seed if attempt == 0 else np.random.SeedSequence([seed, attempt]))


def _render(params, sink) -> tuple[Renderer, list[str], float, float]:
    """Render every stream in one pass into `sink`; return (renderer, streams, green artery SNR, green vein SNR)."""
    has_di = params.streams.has_depth_ir
    streams = ["rgb", "ir", "depth"] if has_di else ["rgb"]
    fps = params.video.fps
    # The sink decides which trace the video is rendered from (FolderSink: the CSV read back off disk).
    trace = sink.begin(generate_trace(params.trace), streams, params.video.n_frames, params.video.frame_size, fps)
    r = Renderer(params, trace)
    art_mask, vein_mask = r.ids == 1, r.ids == 2
    g_art, g_vein = np.empty(r.n_frames), np.empty(r.n_frames)
    for i in range(r.n_frames):
        rgb = r.frame(i)
        g = rgb[..., 1].astype(np.float64)
        g_art[i], g_vein[i] = g[art_mask].mean(), g[vein_mask].mean()
        # Arguments evaluate left to right, keeping the sensor noise draws in rgb -> ir -> depth order.
        sink.frame(rgb, r.ir_frame(i) if has_di else None, r.depth_frame(i) if has_di else None)
    hr_hz = params.trace.heart_rate_bpm / 60.0
    return r, streams, cardiac_snr(g_art, fps, hr_hz), cardiac_snr(g_vein, fps, hr_hz)


def generate_sample(config: GeneratorConfig, seed: int, out_dir: Path, preset: str = "custom", *,
                    zarr: bool = False) -> dict:
    """Render one sample into `out_dir` (or, with `zarr`, the store `{out_dir}.zarr`) and return its metadata
    dict. Guarantees a testable green-channel pulse: both vessels are checked and, if either is under
    MIN_VISIBLE_SNR, the sample is redrawn from an independent sub-stream of `seed`, up to
    MAX_VISIBILITY_ATTEMPTS times. On failure nothing is left behind."""
    validate(config)
    if zarr:
        check_config(config)
    sink = ZarrSink(out_dir) if zarr else FolderSink(out_dir)
    try:
        last = (0.0, 0.0)
        for attempt in range(MAX_VISIBILITY_ATTEMPTS):
            params = sample(config, _attempt_rng(seed, attempt))
            r, streams, artery_snr, vein_snr = _render(params, sink)
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
                "abp_site": params.trace.abp_site,
                "abp_site_delay_s": params.trace.abp_site_delay_s,
                "r_to_abp_foot_s": params.trace.r_to_abp_foot_s,
                "r_to_ppg_foot_s": params.trace.r_to_ppg_foot_s,
            },
            "rgbid_streams": {str(i): s for i, s in enumerate(streams)},
            "depth_units_mm": DEPTH_UNITS_MM,
            "vessel_ids": {"file": "vessel_ids.npy", "0": "background", "1": "artery", "2": "vein"},
            "visibility": {"channel": "G", "min_snr": MIN_VISIBLE_SNR, "attempts": attempt + 1,
                           "artery_snr": artery_snr, "vein_snr": vein_snr},
        }
        sink.commit(r.ids, meta)
    except BaseException:
        sink.discard()
        raise
    return meta


def _one(args) -> tuple[int, Exception | None]:
    config, index, seed, out_dir, preset, zarr = args
    try:
        generate_sample(config, seed, out_dir, preset, zarr=zarr)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller; the sink already removed its output
        return index, e


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:               # noqa: BLE001 - not a git checkout
        return None


def generate_dataset(config: GeneratorConfig, out_root: Path, n: int, start: int, base_seed: int,
                     preset: str, overrides: list[str], jobs: int = 1, *,
                     zarr: bool = False) -> list[tuple[int, Exception | None]]:
    """Generate samples start..start+n-1 (seed = base_seed + i) and write dataset.json. With `zarr`, each
    sample is a store `{i}.zarr` in `out_root` instead of a folder."""
    validate(config)
    if zarr:
        check_config(config)
    else:
        require_ffmpeg()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_args = [(config, i, base_seed + i, out_root / str(i), preset, zarr) for i in range(start, start + n)]
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
        "format": "zarr" if zarr else "folder",
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
