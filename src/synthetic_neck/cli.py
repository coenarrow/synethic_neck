"""`synthetic-neck` command line: generate a dataset of samples from a priors file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_PRIORS
from .frames.backend import DEVICES
from .store import CHUNK_FRAMES, DEFAULT_CLEVEL


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synthetic-neck", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="render a dataset of synthetic samples, one zarr store each")
    g.add_argument("--priors", type=Path, default=DEFAULT_PRIORS, help="YAML priors file (default priors/base.yaml)")
    g.add_argument("--out", type=Path, default=Path("data/synthetic_necks"))
    g.add_argument("--n", type=int, default=5)
    g.add_argument("--start", type=int, default=1)
    g.add_argument("--seed", type=int, default=2026, help="base seed; sample i uses seed+i")
    g.add_argument("--jobs", type=int, default=1, help="samples rendered concurrently; on a GPU, 1 is usually best")
    g.add_argument("--device", choices=DEVICES, default="auto",
                   help="where frames render: cpu is numpy; cuda and mps need the gpu extra; auto takes a GPU if there is one")
    g.add_argument("--batch", type=int, default=CHUNK_FRAMES,
                   help=f"frames rendered together; the store's chunk of {CHUNK_FRAMES} writes whole chunks")
    g.add_argument("--size", type=int, default=None,
                   help="delivered frame size in pixels; the native 650 px crop when omitted")
    g.add_argument("--compression", type=int, default=DEFAULT_CLEVEL, choices=range(0, 10), metavar="{0..9}",
                   help=f"zstd level of the stored frames: 0 none, 1 fastest, 9 smallest (default {DEFAULT_CLEVEL}); "
                        "the noise keeps the frames near half size at any level, and 9 writes 18 times slower than 5")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from .generate import generate_dataset
        results = generate_dataset(args.out, n=args.n, start=args.start, base_seed=args.seed, jobs=args.jobs,
                                   output_px=args.size, priors_path=args.priors, device=args.device,
                                   batch=args.batch, clevel=args.compression)
        failed = [i for i, e in results if e is not None]
        print(f"wrote {len(results) - len(failed)}/{len(results)} samples to {args.out}")
        return 1 if failed else 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
