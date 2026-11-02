"""Experiment configuration schema and loader.

The schema is validated with pydantic. Validation is purely structural and
numerical: it never touches the filesystem, so an example configuration can be
checked before any NISAR product has been downloaded. Path existence is a
separate concern handled at experiment run time (not yet implemented).

Solver parameter names mirror the keyword arguments of the dwave-samplers
classes they target (verified against dwave-samplers 1.8.0):

* ``SimulatedAnnealingSampler.sample``: ``num_reads``, ``num_sweeps``,
  ``beta_range``, ``beta_schedule_type``, ``seed``.
* ``PathIntegralAnnealingSampler.sample``: ``num_reads``,
  ``num_sweeps_per_beta``, ``beta_schedule_type="custom"``, ``Hp_field``,
  ``Hd_field``, ``seed``.

The SQA section deliberately does not expose ``Gamma`` or the linear/geometric
schedules: in dwave-samplers 1.8.0 those schedules set the transverse field to
zero at every step (``Gamma`` is overwritten), which is classical annealing.
See ``insar_unwrap.solvers.annealing``.

Both samplers reject seeds outside ``[0, 2**31 - 1]`` although their error
message says ``2^32 - 1``; the schema uses the bound the code enforces.

Only the subset we intend to use is modelled here. Unknown keys are rejected so
that typos do not silently fall back to sampler defaults.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PositiveInt = Annotated[int, Field(ge=1)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]

SEED_MAX = 2**31 - 1
"""Largest seed accepted by dwave-samplers 1.8.0 (checked: ``0 <= seed < 2**31``)."""


class _Strict(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class ExperimentSection(_Strict):
    """Top-level experiment identity and reproducibility settings."""

    name: str = Field(min_length=1)
    seed: Annotated[int, Field(ge=0, le=SEED_MAX)] = 0
    output_dir: Path = Path("results")


class Hdf5Layers(_Strict):
    """HDF5 dataset paths inside a GUNW product.

    The defaults in the example config are placeholders and must be confirmed
    against a real product before use (see docs/experiment-plan.md).
    """

    wrapped: str = Field(min_length=1)
    coherence: str = Field(min_length=1)
    mask: str | None = None


class CropSection(_Strict):
    """Crop origin on the native grid. Sizes come from ``problem.sizes``."""

    row0: Annotated[int, Field(ge=0)] = 0
    col0: Annotated[int, Field(ge=0)] = 0


class MaskSection(_Strict):
    """How invalid pixels are decided."""

    use_valid_mask: bool = True
    coherence_threshold: UnitInterval = 0.0


class InputSection(_Strict):
    """Input data description.

    ``kind == "synthetic"`` is for implementation checks; ``kind == "nisar_gunw"``
    is for real-data evaluation. The two must never be mixed in one report.
    """

    kind: Literal["nisar_gunw", "synthetic"]
    path: Path | None = None
    frequency: Literal["A", "B"] = "A"
    polarization: Literal["HH", "HV", "VH", "VV"] = "HH"
    layers: Hdf5Layers | None = None
    crop: CropSection = CropSection()
    mask: MaskSection = MaskSection()

    @model_validator(mode="after")
    def _require_layers_for_gunw(self) -> InputSection:
        if self.kind == "nisar_gunw":
            if self.path is None:
                raise ValueError("input.path is required when input.kind == 'nisar_gunw'")
            if self.layers is None:
                raise ValueError("input.layers is required when input.kind == 'nisar_gunw'")
        return self


class ReferenceSection(_Strict):
    """Gauge fixing: which pixel(s) pin the integer label.

    ``per_component`` means one reference per connected component of the valid
    mask; ``pixel`` means a single explicit pixel (only meaningful if the valid
    graph is connected).
    """

    mode: Literal["pixel", "per_component"] = "per_component"
    pixel: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None = None

    @model_validator(mode="after")
    def _pixel_only_with_pixel_mode(self) -> ReferenceSection:
        if self.mode == "pixel" and self.pixel is None:
            raise ValueError("problem.reference.pixel is required when mode == 'pixel'")
        if self.mode != "pixel" and self.pixel is not None:
            raise ValueError("problem.reference.pixel is only allowed when mode == 'pixel'")
        return self


class ProblemSection(_Strict):
    """QUBO problem definition."""

    sizes: list[PositiveInt] = Field(min_length=1)
    label_min: int
    label_max: int
    encoding: Literal["binary", "one_hot"] = "binary"
    edge_weighting: Literal["uniform", "coherence"] = "coherence"
    reference: ReferenceSection = ReferenceSection()

    @model_validator(mode="after")
    def _check_ranges(self) -> ProblemSection:
        if self.label_min >= self.label_max:
            raise ValueError("problem.label_min must be strictly less than problem.label_max")
        if self.sizes != sorted(self.sizes):
            raise ValueError("problem.sizes must be in ascending order")
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("problem.sizes must not contain duplicates")
        return self

    @property
    def num_labels(self) -> int:
        """Number of integer labels per pixel."""
        return self.label_max - self.label_min + 1

    @property
    def bits_per_pixel(self) -> int:
        """Binary variables needed per pixel for the chosen encoding."""
        if self.encoding == "one_hot":
            return self.num_labels
        return max(1, math.ceil(math.log2(self.num_labels)))


def _check_beta_range(beta_range: tuple[float, float] | None, name: str) -> None:
    """Validate an optional (beta_min, beta_max) pair of inverse temperatures."""
    if beta_range is None:
        return
    lo, hi = beta_range
    if lo <= 0 or hi <= 0:
        raise ValueError(f"{name} values must be positive")
    if lo >= hi:
        raise ValueError(f"{name} must satisfy beta_min < beta_max")


class SaSolver(_Strict):
    """Classical simulated annealing (dwave.samplers.SimulatedAnnealingSampler)."""

    enabled: bool = True
    num_reads: PositiveInt = 10
    num_sweeps: PositiveInt = 1000
    beta_range: tuple[float, float] | None = None
    beta_schedule_type: Literal["linear", "geometric"] = "geometric"

    @model_validator(mode="after")
    def _check_beta(self) -> SaSolver:
        _check_beta_range(self.beta_range, "solvers.sa.beta_range")
        return self


class SqaSchedule(_Strict):
    """Fixed-temperature transverse-field schedule.

    Step ``i`` uses ``Hp_field[i] = beta`` and ``Hd_field[i] = beta * Gamma_i``
    with ``Gamma_i`` linear from ``gamma_start`` to ``gamma_end``. ``beta`` and
    ``Gamma`` are in units of the BQM energy (the sampler converts the BQM to
    Ising form; energies are unchanged by that conversion). The defaults are
    untuned starting values, not calibrated settings.
    """

    beta: PositiveFloat = 10.0
    gamma_start: PositiveFloat = 3.0
    gamma_end: NonNegativeFloat = 0.0
    num_steps: PositiveInt = 1000

    @model_validator(mode="after")
    def _decreasing(self) -> SqaSchedule:
        if self.gamma_end >= self.gamma_start:
            raise ValueError("solvers.sqa.schedule.gamma_end must be < gamma_start")
        return self


class SqaSolver(_Strict):
    """Simulated quantum annealing (dwave.samplers.PathIntegralAnnealingSampler).

    Always run with ``beta_schedule_type="custom"``. Either ``schedule``
    (generated) or explicit ``Hp_field``/``Hd_field`` lists (same length,
    non-negative, scaled by inverse temperature) define the anneal. Explicit
    fields require that ``schedule`` is not given, and ``Hd_field`` must be
    positive somewhere; an all-zero transverse field is classical annealing.
    Total sweeps per read are ``len(schedule) * num_sweeps_per_beta``.
    """

    enabled: bool = True
    num_reads: PositiveInt = 10
    num_sweeps_per_beta: PositiveInt = 1
    schedule: SqaSchedule = SqaSchedule()
    Hp_field: list[NonNegativeFloat] | None = None
    Hd_field: list[NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> SqaSolver:
        if (self.Hp_field is None) != (self.Hd_field is None):
            raise ValueError("solvers.sqa.Hp_field and Hd_field must be given together")
        if self.Hp_field is not None and self.Hd_field is not None:
            if "schedule" in self.model_fields_set:
                raise ValueError("give either solvers.sqa.schedule or Hp_field/Hd_field, not both")
            if len(self.Hp_field) != len(self.Hd_field):
                raise ValueError("solvers.sqa.Hp_field and Hd_field must have the same length")
            if not self.Hp_field:
                raise ValueError("solvers.sqa.Hp_field must not be empty")
            if max(self.Hd_field) <= 0:
                raise ValueError(
                    "solvers.sqa.Hd_field is zero everywhere; that is classical annealing, not SQA"
                )
        return self

    @property
    def total_sweeps(self) -> int:
        """Sweeps per read implied by the schedule."""
        steps = len(self.Hp_field) if self.Hp_field is not None else self.schedule.num_steps
        return steps * self.num_sweeps_per_beta


class SnaphuSolver(_Strict):
    """External SNAPHU executable (reference unwrapper, not a QUBO solver)."""

    enabled: bool = False
    executable: str = "snaphu"
    config_template: Path | None = None


class SolversSection(_Strict):
    """Which solvers to run."""

    sa: SaSolver = SaSolver()
    sqa: SqaSolver = SqaSolver()
    snaphu: SnaphuSolver = SnaphuSolver()

    @model_validator(mode="after")
    def _at_least_one(self) -> SolversSection:
        if not (self.sa.enabled or self.sqa.enabled or self.snaphu.enabled):
            raise ValueError("at least one solver must be enabled")
        return self


class ExperimentConfig(_Strict):
    """Root configuration object."""

    experiment: ExperimentSection
    input: InputSection
    problem: ProblemSection
    solvers: SolversSection = SolversSection()

    @model_validator(mode="after")
    def _cross_section(self) -> ExperimentConfig:
        if self.problem.edge_weighting == "coherence" and self.input.kind == "nisar_gunw":
            layers = self.input.layers
            if layers is None or not layers.coherence:
                raise ValueError("coherence weighting requires input.layers.coherence")
        if self.input.kind == "synthetic" and self.input.path is not None:
            raise ValueError("input.path must be omitted when input.kind == 'synthetic'")
        return self


class ConfigError(Exception):
    """Raised when a configuration file cannot be parsed or fails validation."""


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, is not a mapping,
            or fails schema validation. The message lists every offending field.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError("configuration is invalid:\n  " + "\n  ".join(lines)) from exc
