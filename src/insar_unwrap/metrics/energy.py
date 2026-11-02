"""Energy statistics over reads and time-to-solution.

Energies are BQM energies as reported by the sampler. For binary encoding they
equal the integer objective; for one-hot encoding they include penalty terms of
infeasible reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyStats:
    """Summary of the energies of independent reads.

    Attributes:
        num_reads: Number of reads (``num_occurrences`` expanded).
        best: Lowest energy.
        mean: Mean energy.
        std: Sample standard deviation (``ddof=1``; 0 for a single read).
        median: Median energy.
        worst: Highest energy.
        success_fraction: Fraction of reads with ``energy <= target + atol``;
            ``None`` if no target was given.
    """

    num_reads: int
    best: float
    mean: float
    std: float
    median: float
    worst: float
    success_fraction: float | None


def _read_energies(energies: object) -> np.ndarray:
    record = getattr(energies, "record", None)
    if record is not None:
        return np.repeat(np.asarray(record.energy, dtype=float), record.num_occurrences)
    return np.asarray(energies, dtype=float).reshape(-1)


def energy_stats(
    energies: object,
    *,
    target: float | None = None,
    atol: float = 1e-9,
) -> EnergyStats:
    """Summarize energies over reads.

    Args:
        energies: A ``dimod.SampleSet`` (rows are expanded by
            ``num_occurrences``) or a 1-D array of per-read energies.
        target: Optional target energy, e.g. the known optimum ``E* = 0`` of a
            residue-free synthetic field under the free gauge.
        atol: Absolute tolerance for reaching ``target``.

    Returns:
        Energy statistics.

    Raises:
        ValueError: If there are no energies or some are non-finite.
    """
    e = _read_energies(energies)
    if e.size == 0:
        raise ValueError("no energies")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must be finite")
    success = None if target is None else float(np.mean(e <= target + atol))
    return EnergyStats(
        num_reads=int(e.size),
        best=float(e.min()),
        mean=float(e.mean()),
        std=float(e.std(ddof=1)) if e.size > 1 else 0.0,
        median=float(np.median(e)),
        worst=float(e.max()),
        success_fraction=success,
    )


def time_to_solution(
    time_per_read: float,
    success_probability: float,
    confidence: float = 0.99,
) -> float:
    """Expected time to reach the target at least once with given confidence.

    ``TTS = t * ln(1 - confidence) / ln(1 - p)``, with ``TTS = t`` once a
    single read already succeeds with probability ``>= confidence``.

    Args:
        time_per_read: Time of one read, e.g. CPU time of a ``sample()`` call
            divided by ``num_reads``. The caller decides which clock.
        success_probability: Estimated per-read success probability ``p``.
        confidence: Required probability of at least one success.

    Returns:
        Time to solution in the unit of ``time_per_read``; ``inf`` if ``p == 0``.

    Raises:
        ValueError: On a negative time or probabilities outside their ranges.
    """
    if time_per_read < 0:
        raise ValueError("time_per_read must be non-negative")
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if success_probability >= confidence:
        return float(time_per_read)
    if success_probability == 0.0:
        return math.inf
    return float(time_per_read * math.log1p(-confidence) / math.log1p(-success_probability))
