"""InSAR phase unwrapping benchmark: QUBO formulation solved with SA and SQA.

This package is currently a skeleton. Only configuration validation and
environment diagnostics are implemented. See docs/experiment-plan.md for the
planned module layout (io, qubo, solvers, metrics).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("insar-unwrap-annealing")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
