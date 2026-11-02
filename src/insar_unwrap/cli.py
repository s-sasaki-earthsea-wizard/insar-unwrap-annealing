"""Command-line entry point for ``insar-unwrap``.

Implemented subcommands:

* ``validate-config``: parse and validate an experiment YAML file.
* ``doctor``: report the local environment (versions, samplers, SNAPHU).

Unwrapping and benchmarking are intentionally not exposed yet; they will be
added once the QUBO formulation has been verified on synthetic data.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from insar_unwrap import __version__
from insar_unwrap.config import ConfigError, load_config
from insar_unwrap.doctor import format_report, has_required_failure, run_checks


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="insar-unwrap",
        description=(
            "InSAR phase unwrapping benchmark (QUBO + SA/SQA vs SNAPHU). "
            "Skeleton release: only configuration validation and diagnostics are available."
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
