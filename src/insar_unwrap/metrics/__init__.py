"""Solution metrics: gauge-aware label agreement, bound saturation, energy statistics."""

from insar_unwrap.metrics.energy import EnergyStats, energy_stats, time_to_solution
from insar_unwrap.metrics.labels import (
    BoundSaturation,
    ModuloAgreement,
    agreement_modulo_constant,
    bound_saturation,
)

__all__ = [
    "BoundSaturation",
    "EnergyStats",
    "ModuloAgreement",
    "agreement_modulo_constant",
    "bound_saturation",
    "energy_stats",
    "time_to_solution",
]
