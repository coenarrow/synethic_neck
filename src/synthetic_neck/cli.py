"""`synthetic-neck` command line: generate | inspect | calibrate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, GeneratorConfig, apply_override, validate
from .presets import PRESETS, get_preset


def build_config(preset: str, sets: list[str]) -> GeneratorConfig:
    cfg = get_preset(preset)
    for item in sets:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, text = item.split("=", 1)
        cfg = apply_override(cfg, key.strip(), text.strip())
    validate(cfg)
    return cfg


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synthetic-neck", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="render a dataset of synthetic samples")
    g.add_argument("--preset", choices=sorted(PRESETS), default="lesson")
    g.add_argument("--set", dest="sets", action="append", default=[], metavar="block.field=value",
                   help="override a config field; Range as lo,hi or v; Choice as a|b|c")
    g.add_argument("--out", type=Path, default=Path("data/synthetic_necks"))
    g.add_argument("--n", type=int, default=5)
    g.add_argument("--start", type=int, default=1)
    g.add_argument("--seed", type=int, default=2026, help="base seed; sample i uses seed+i")
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--zarr", action="store_true",
                   help="write one cache-contract zarr store per sample ({i}.zarr) instead of the folder layout")

    i = sub.add_parser("inspect", help="write fft_maps.png and print vessel power/phase per sample")
    i.add_argument("--root", type=Path, default=Path("data/synthetic_necks"))
    i.add_argument("--samples", nargs="*")

    c = sub.add_parser("calibrate", help="mine the Neckflix dataset for priors (offline)")
    c.add_argument("--root", type=Path, required=True, help="Neckflix dataset directory")
    c.add_argument("--out", type=Path, default=Path("priors/neckflix.json"))
    c.add_argument("--n-recordings", type=int, default=30)
    c.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            from .generate import generate_dataset
            cfg = build_config(args.preset, args.sets)
            results = generate_dataset(cfg, args.out, n=args.n, start=args.start, base_seed=args.seed,
                                       preset=args.preset, overrides=args.sets, jobs=args.jobs, zarr=args.zarr)
            failed = [i for i, e in results if e is not None]
            print(f"wrote {len(results) - len(failed)}/{len(results)} samples to {args.out}")
            return 1 if failed else 0
        if args.command == "inspect":
            from .inspect import inspect_root
            inspect_root(args.root, args.samples)
            return 0
        if args.command == "calibrate":
            from .calibrate import run_calibration
            run_calibration(args.root, args.out, n_recordings=args.n_recordings, seed=args.seed)
            print(f"wrote {args.out}")
            return 0
    except (ConfigError, FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
