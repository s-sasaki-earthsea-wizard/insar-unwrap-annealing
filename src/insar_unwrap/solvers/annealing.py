"""Run one BQM through the local SA and SQA samplers of dwave-samplers.

Both adapters take the BQM unchanged and return the raw sample set plus wall
and CPU time of the ``sample()`` call. Nothing here interprets labels; decoding
belongs to :class:`insar_unwrap.qubo.UnwrapQubo`.

SQA schedule (verified against dwave-samplers 1.8.0 source and by experiment):
``PathIntegralAnnealingSampler`` overwrites its ``Gamma`` argument with
``Hd_field[i]`` at every schedule step, and for ``beta_schedule_type`` of
``"linear"``/``"geometric"`` it sets ``Hd_field`` to zeros. With those schedules
no worldline breaks are ever proposed and each update is a classical
heat-bath spin flip, i.e. classical annealing. :func:`run_sqa` therefore always
passes ``beta_schedule_type="custom"`` with an explicit, non-zero transverse
field schedule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import dimod
import numpy as np
from dwave.samplers import PathIntegralAnnealingSampler, SimulatedAnnealingSampler

from insar_unwrap.config import SEED_MAX, SaSolver, SqaSchedule, SqaSolver


@dataclass(frozen=True)
class SamplerRun:
    """Outcome of one ``sample()`` call.

    Attributes:
        solver: ``"sa"`` or ``"sqa"``.
        sampler: Fully qualified sampler class name.
        params: Keyword arguments passed to ``sample()`` (arrays as lists).
        sampleset: The returned sample set (energies include the BQM offset).
        wall_time_s: ``time.perf_counter`` duration of the call.
        cpu_time_s: ``time.process_time`` duration of the call (this process).
    """

    solver: str
    sampler: str
    params: dict[str, Any]
    sampleset: dimod.SampleSet
    wall_time_s: float
    cpu_time_s: float

    @property
    def energies(self) -> np.ndarray:
        """Energy of every read, in sample order."""
        return np.asarray(self.sampleset.record.energy, dtype=float)

    @property
    def best_energy(self) -> float:
        """Lowest energy over all reads."""
        return float(self.energies.min())


def sqa_schedule(schedule: SqaSchedule) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-temperature schedule with a linearly decreasing transverse field.

    The sampler's scaled Hamiltonian at step ``i`` is
    ``Hp_field[i] * H_problem - Hd_field[i] * sum_j sigma^x_j``. Here
    ``Hp_field[i] = beta`` and ``Hd_field[i] = beta * Gamma_i`` with
    ``Gamma_i`` linear from ``gamma_start`` to ``gamma_end``.

    Args:
        schedule: Validated schedule parameters.

    Returns:
        ``(Hp_field, Hd_field)``, each of length ``schedule.num_steps``.
    """
    gamma = np.linspace(schedule.gamma_start, schedule.gamma_end, schedule.num_steps)
    hp = np.full(schedule.num_steps, schedule.beta)
    return hp, schedule.beta * gamma


def _check_seed(seed: int) -> int:
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"seed must be in [0, {SEED_MAX}] for dwave-samplers, got {seed}")
    return int(seed)


def _timed(solver: str, sampler: Any, bqm: dimod.BinaryQuadraticModel, params: dict) -> SamplerRun:
    t0, c0 = time.perf_counter(), time.process_time()
    sampleset = sampler.sample(bqm, **params)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    recorded = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}
    return SamplerRun(
        solver=solver,
        sampler=f"{type(sampler).__module__}.{type(sampler).__name__}",
        params=recorded,
        sampleset=sampleset,
        wall_time_s=wall,
        cpu_time_s=cpu,
    )


def run_sa(bqm: dimod.BinaryQuadraticModel, cfg: SaSolver, seed: int) -> SamplerRun:
    """Sample ``bqm`` with ``SimulatedAnnealingSampler`` (Metropolis, sequential sweeps).

    Args:
        bqm: The problem; not modified.
        cfg: SA settings.
        seed: PRNG seed in ``[0, 2**31 - 1]``.

    Returns:
        The run record.
    """
    params: dict[str, Any] = {
        "num_reads": cfg.num_reads,
        "num_sweeps": cfg.num_sweeps,
        "beta_schedule_type": cfg.beta_schedule_type,
        "seed": _check_seed(seed),
    }
    if cfg.beta_range is not None:
        params["beta_range"] = list(cfg.beta_range)
    return _timed("sa", SimulatedAnnealingSampler(), bqm, params)


def run_sqa(bqm: dimod.BinaryQuadraticModel, cfg: SqaSolver, seed: int) -> SamplerRun:
    """Sample ``bqm`` with ``PathIntegralAnnealingSampler`` on a custom schedule.

    Args:
        bqm: The problem; not modified.
        cfg: SQA settings. Explicit ``Hp_field``/``Hd_field`` take precedence
            over ``cfg.schedule``.
        seed: PRNG seed in ``[0, 2**31 - 1]``.

    Returns:
        The run record.

    Raises:
        ValueError: If the transverse field is zero at every step (that would
            be classical annealing, not SQA) or the seed is out of range.
    """
    if cfg.Hp_field is not None and cfg.Hd_field is not None:
        hp = np.asarray(cfg.Hp_field, dtype=float)
        hd = np.asarray(cfg.Hd_field, dtype=float)
    else:
        hp, hd = sqa_schedule(cfg.schedule)
    if not np.any(hd > 0):
        raise ValueError("Hd_field is zero everywhere; that is classical annealing, not SQA")
    params: dict[str, Any] = {
        "num_reads": cfg.num_reads,
        "num_sweeps_per_beta": cfg.num_sweeps_per_beta,
        "beta_schedule_type": "custom",
        "Hp_field": hp,
        "Hd_field": hd,
        "seed": _check_seed(seed),
    }
    return _timed("sqa", PathIntegralAnnealingSampler(), bqm, params)
