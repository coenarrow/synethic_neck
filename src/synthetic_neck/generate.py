"""Generate one sample or a whole dataset: draw, step the frames on the shared clock, and write the store."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from . import __version__
from .config import DEFAULT_PRIORS, Config, load
from .sample import describe, draw_sample, frame_at, stored_traces
from .store import ZarrSink


def generate_sample(cfg: Config, seed: int, out_dir: Path, output_px: int | None = None,
                    priors: str | None = None, progress: bool = True) -> dict:
    """Draw the sample for `seed`, render its frames and write `{out_dir}.zarr`; return the metadata. On failure
    nothing is left behind. With `progress`, a tqdm bar over the frames."""
    sink = ZarrSink(out_dir)
    try:
        sample = draw_sample(cfg, seed, output_px)
        sink.begin(sample.n_frames, sample.output_px, sample.sample_rate_hz)
        frames = tqdm(sample.frame_times_s, desc=f"sample {sink.name} frames", unit="frame", leave=False,
                      disable=not progress)
        for t in frames:
            f = frame_at(sample, float(t))
            sink.frame(f.rgb, f.ir, f.depth_mm)
        meta = describe(sample)
        meta["version"], meta["priors"] = __version__, priors
        sink.commit(stored_traces(sample), sample.labels, sample.neck_mask, meta)
    except BaseException:
        sink.discard()
        raise
    return meta


def _one(args) -> tuple[int, Exception | None]:
    priors_path, index, seed, out_dir, output_px, progress = args
    try:
        generate_sample(load(priors_path), seed, out_dir, output_px, str(priors_path), progress)
        return index, None
    except Exception as e:          # noqa: BLE001 - reported to the caller; the sink already removed its output
        return index, e


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:               # noqa: BLE001 - not a git checkout
        return None


def generate_dataset(out_root: Path, n: int, start: int = 1, base_seed: int = 2026, jobs: int = 1,
                     output_px: int | None = None, priors_path: Path = DEFAULT_PRIORS,
                     progress: bool = True) -> list[tuple[int, Exception | None]]:
    """Generate samples start..start+n-1 (seed = base_seed + i) as `{i}.zarr` in `out_root`, with dataset.json.
    With `progress`, tqdm bars over the samples and, when `jobs` is 1, the frames of the current sample."""
    priors_path = Path(priors_path)
    load(priors_path)                                      # fail before any work on a bad priors file
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # The frame bar is shown only when samples run one at a time; with workers, the samples bar alone.
    jobs_args = [(priors_path, i, base_seed + i, out_root / str(i), output_px, progress and jobs <= 1) for i in range(start, start + n)]
    samples = tqdm(total=n, desc="samples", unit="sample", disable=not progress)
    if jobs <= 1:
        results = []
        for a in jobs_args:
            results.append(_one(a))
            samples.update()
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_one, a) for a in jobs_args]
            results = []
            for fut in as_completed(futures):
                results.append(fut.result())
                samples.update()
            results.sort(key=lambda r: r[0])
    samples.close()
    index = {
        "priors": str(priors_path),
        "priors_values": yaml.safe_load(priors_path.read_text()),
        "output_px": output_px,
        "base_seed": base_seed, "start": start, "n": n,
        "version": __version__, "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "failed": [{"index": i, "seed": base_seed + i} for i, e in results if e is not None],
    }
    (out_root / "dataset.json").write_text(json.dumps(index, indent=2))
    for i, e in results:
        if e is not None:
            print(f"sample {i} (seed {base_seed + i}) failed: {e!r}", file=sys.stderr)
    return results
