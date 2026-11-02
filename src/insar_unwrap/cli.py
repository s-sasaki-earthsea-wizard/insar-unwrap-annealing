"""Command-line entry point for ``insar-unwrap``.

Implemented subcommands:

* ``validate-config``: parse and validate an experiment YAML file.
* ``doctor``: report the local environment (versions, samplers, SNAPHU).
* ``datasets`` / ``fetch-sample``: list and download approved sample datasets.
* ``inspect``: list layers, grids and mask attributes of a GUNW file.
* ``crop-summary``: read the configured crop and build its BQM (no solving),
  as a wiring check from configuration to QUBO.

Unwrapping runs and benchmarks are intentionally not exposed yet.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from insar_unwrap import __version__
from insar_unwrap.config import ConfigError, load_config
from insar_unwrap.doctor import format_report, has_required_failure, run_checks
from insar_unwrap.io.datasets import (
    DEFAULT_DEST,
    DEFAULT_REGISTRY,
    DatasetError,
    fetch_dataset,
    load_registry,
)
from insar_unwrap.io.gunw import GunwError, describe, read_crop
from insar_unwrap.metrics.descriptors import mean_fringe_rate, tree_label_span
from insar_unwrap.phase import residues
from insar_unwrap.qubo import build_unwrap_bqm, edge_weights


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="insar-unwrap",
        description=(
            "InSAR phase unwrapping benchmark (QUBO + SA/SQA vs SNAPHU). "
            "Configuration, data access and QUBO wiring checks; no benchmark runs yet."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate-config",
        help="validate an experiment configuration file (no data access)",
    )
    p_val.add_argument("--config", required=True, help="path to the YAML configuration")
    p_val.set_defaults(func=_cmd_validate_config)

    p_doc = sub.add_parser("doctor", help="report Python, package, sampler and SNAPHU status")
    p_doc.add_argument(
        "--snaphu-executable",
        default="snaphu",
        help="name or path of the SNAPHU binary to look for (default: snaphu)",
    )
    p_doc.set_defaults(func=_cmd_doctor)

    p_ds = sub.add_parser("datasets", help="list approved sample datasets")
    p_ds.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="dataset registry YAML")
    p_ds.set_defaults(func=_cmd_datasets)

    p_fetch = sub.add_parser(
        "fetch-sample",
        help="download an approved dataset into data/raw/NAME and verify it",
    )
    p_fetch.add_argument("name", help="dataset name from the registry")
    p_fetch.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="dataset registry YAML")
    p_fetch.add_argument("--dest", default=str(DEFAULT_DEST), help="root directory for raw data")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_insp = sub.add_parser("inspect", help="list layers, grids and masks of a GUNW file")
    p_insp.add_argument("file", help="GUNW HDF5 file")
    p_insp.add_argument("--root", default="/science/LSAR/GUNW/grids", help="group to list")
    p_insp.set_defaults(func=_cmd_inspect)

    p_crop = sub.add_parser(
        "crop-summary",
        help="read the configured crop and build its BQM (no solving)",
    )
    p_crop.add_argument("--config", required=True, help="path to the YAML configuration")
    p_crop.add_argument(
        "--size", type=int, default=None, help="crop size (default: smallest problem size)"
    )
    p_crop.set_defaults(func=_cmd_crop_summary)
    return parser


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    enabled = [n for n in ("sa", "sqa", "snaphu") if getattr(cfg.solvers, n).enabled]
    print(f"OK: {args.config}")
    print(f"  experiment: {cfg.experiment.name} (seed={cfg.experiment.seed})")
    print(f"  input: kind={cfg.input.kind}")
    print(
        f"  problem: sizes={cfg.problem.sizes}, labels=[{cfg.problem.label_min}, "
        f"{cfg.problem.label_max}] -> {cfg.problem.num_labels} labels, "
        f"{cfg.problem.bits_per_pixel} bit(s)/pixel ({cfg.problem.encoding})"
    )
    print(f"  solvers enabled: {', '.join(enabled)}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    results = run_checks(snaphu_executable=args.snaphu_executable)
    print(format_report(results))
    return 1 if has_required_failure(results) else 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    try:
        registry = load_registry(args.registry)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, entry in registry.datasets.items():
        login = "login" if entry.requires_login else "no login"
        print(f"{name}  ({entry.size_bytes / 2**20:,.0f} MiB, {login})")
        print(f"  {entry.description}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        result = fetch_dataset(args.name, load_registry(args.registry), Path(args.dest))
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{result.status}: {result.path}")
    print(f"provenance: {result.provenance}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        info = describe(args.file, args.root)
    except (GunwError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(yaml.safe_dump(info, sort_keys=False, width=100), end="")
    return 0


def _percentiles(x: np.ndarray, q: Sequence[float]) -> list[float] | None:
    return [round(float(v), 4) for v in np.percentile(x, q)] if x.size else None


def _cmd_crop_summary(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    inp, prob = cfg.input, cfg.problem
    if inp.kind != "nisar_gunw" or inp.path is None or inp.layers is None:
        print("error: crop-summary needs input.kind == 'nisar_gunw'", file=sys.stderr)
        return 1
    size = args.size if args.size is not None else prob.sizes[0]
    try:
        crop = read_crop(
            inp.path,
            wrapped=inp.layers.wrapped,
            coherence=inp.layers.coherence,
            mask=inp.layers.mask if inp.mask.use_valid_mask else None,
            row0=inp.crop.row0,
            col0=inp.crop.col0,
            rows=size,
            cols=size,
            coherence_threshold=inp.mask.coherence_threshold,
            exclude_water=inp.mask.exclude_water,
            looks=inp.multilook.looks,
            min_valid_fraction=inp.multilook.min_valid_fraction,
        )
        weights = edge_weights(
            prob.edge_weighting,
            crop.valid,
            crop.coherence,
            normalization=prob.weight_normalization,
        )
        qubo = build_unwrap_bqm(
            crop.wrapped,
            label_min=prob.label_min,
            label_max=prob.label_max,
            encoding=prob.encoding,
            valid=crop.valid,
            weight_h=weights.horizontal,
            weight_v=weights.vertical,
            reference_mode=prob.reference.mode,
            reference_pixel=prob.reference.pixel,
        )
    except (GunwError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    valid = crop.valid
    loops = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    res = residues(np.where(valid, crop.wrapped, 0.0))
    rate = mean_fringe_rate(crop.wrapped, valid)
    e = qubo.edges
    span = tree_label_span(valid.size, e.src, e.dst, e.target, valid)
    warnings = []
    if span + 1 > prob.num_labels:
        warnings.append(
            f"tree-integrated label span {span} needs {span + 1} labels; the configured range "
            f"has {prob.num_labels} (rough guide, see insar_unwrap.metrics.descriptors)"
        )
    summary = {
        "source": crop.source,
        "window": {
            "row0": crop.row0,
            "col0": crop.col0,
            "size": size,
            "looks": crop.looks,
            "epsg": crop.epsg,
            "x_first_last": [float(crop.x[0]), float(crop.x[-1])],
            "y_first_last": [float(crop.y[0]), float(crop.y[-1])],
        },
        "valid_fraction": round(float(valid.mean()), 4),
        "coherence_valid_p5_p50_p95": _percentiles(crop.coherence[valid], [5, 50, 95]),
        "look_consistency_valid_p5_p50_p95": (
            _percentiles(crop.consistency[valid], [5, 50, 95]) if crop.looks > 1 else None
        ),
        "residues": {"nonzero": int(np.abs(res[loops]).sum()), "loops": int(loops.sum())},
        "edges_with_wrap": int(np.count_nonzero(qubo.edges.target)),
        "mean_fringes_per_32px": round(rate * 32, 4),
        "mean_fringes_across_crop": round(rate * size, 3),
        "tree_label_span": span,
        "edge_weights": weights.summary(),
        "qubo": qubo.summary(),
        "warnings": warnings,
    }
    print(yaml.safe_dump(summary, sort_keys=False, width=100), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
