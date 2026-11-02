"""Environment diagnostics.

Reports interpreter and package versions, whether the SA and SQA sampler
classes can be imported, and whether the optional SNAPHU executable is on PATH.
It performs no network access and never prints environment variables or
credentials.

Severity model:

* ``required``: a missing or broken item means the Python environment itself is
  unusable (exit status 1).
* ``optional``: the item is a planned external dependency; its absence is
  reported but does not fail the check.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

Severity = Literal["required", "optional"]

REQUIRED_PACKAGES: tuple[str, ...] = ("numpy", "dimod", "dwave-samplers", "pydantic", "pyyaml")
OPTIONAL_PACKAGES: tuple[str, ...] = ("h5py",)

# (label, module, class name, role)
SAMPLER_CLASSES: tuple[tuple[str, str, str, str], ...] = (
    ("SA", "dwave.samplers", "SimulatedAnnealingSampler", "classical simulated annealing"),
    ("SQA", "dwave.samplers", "PathIntegralAnnealingSampler", "simulated quantum annealing"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one diagnostic probe."""

    name: str
    ok: bool
    detail: str
    severity: Severity = "required"

    @property
    def status(self) -> str:
        """Short status token for display."""
        if self.ok:
            return "OK"
        return "MISSING" if self.severity == "optional" else "FAIL"


def check_python() -> CheckResult:
    """Report the running interpreter."""
    return CheckResult(
        "python",
        True,
        f"{platform.python_version()} ({platform.python_implementation()}, {sys.executable})",
    )


def check_package(dist: str, severity: Severity = "required") -> CheckResult:
    """Report the installed version of a distribution."""
    try:
        return CheckResult(dist, True, version(dist), severity)
    except PackageNotFoundError:
        return CheckResult(dist, False, "not installed", severity)


def check_class(label: str, module: str, cls: str, role: str) -> CheckResult:
    """Try to import a sampler class."""
    try:
        mod = importlib.import_module(module)
        obj = getattr(mod, cls)
    except (ImportError, AttributeError) as exc:
        return CheckResult(f"{label} sampler", False, f"{module}.{cls} unavailable: {exc}")
    return CheckResult(f"{label} sampler", True, f"{obj.__module__}.{cls} ({role})")


def check_executable(name: str, severity: Severity = "optional") -> CheckResult:
    """Look for an executable on PATH without running it."""
    found = shutil.which(name)
    if found is None:
        return CheckResult(name, False, "not found on PATH", severity)
    return CheckResult(name, True, found, severity)


def run_checks(snaphu_executable: str = "snaphu") -> list[CheckResult]:
    """Run every diagnostic probe.

    Args:
        snaphu_executable: Name or path of the SNAPHU binary to look for.

    Returns:
        Results in display order.
    """
    results = [check_python()]
    results += [check_package(p) for p in REQUIRED_PACKAGES]
    results += [check_package(p, "optional") for p in OPTIONAL_PACKAGES]
    results += [check_class(*spec) for spec in SAMPLER_CLASSES]
    results.append(check_executable(snaphu_executable))
    return results


def has_required_failure(results: list[CheckResult]) -> bool:
    """Return True if any required check failed."""
    return any(not r.ok and r.severity == "required" for r in results)


def format_report(results: list[CheckResult]) -> str:
    """Render results as an aligned text table."""
    width = max(len(r.name) for r in results)
    lines = [f"{r.name:<{width}}  {r.status:<7} {r.detail}" for r in results]
    if has_required_failure(results):
        lines.append("")
        lines.append("One or more required checks failed; the Python environment is not usable.")
    else:
        missing = [r.name for r in results if not r.ok]
        lines.append("")
        if missing:
            lines.append(f"Required environment OK. Optional items missing: {', '.join(missing)}.")
        else:
            lines.append("All checks passed.")
    return "\n".join(lines)
