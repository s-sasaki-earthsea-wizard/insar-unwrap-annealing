import math

import dimod
import numpy as np
import pytest

from insar_unwrap.metrics import (
    agreement_modulo_constant,
    bound_saturation,
    energy_stats,
    time_to_solution,
)


def test_agreement_removes_one_offset_per_component():
    component = np.array([[0, 0, -1, 1], [0, 0, -1, 1]])
    k_b = np.array([[0, 1, 9, 2], [1, 2, 9, 3]])
    k_a = k_b + np.array([[3, 3, 0, -1], [3, 3, 0, -1]])
    k_a[1, 1] += 1  # one disagreeing pixel in component 0

    result = agreement_modulo_constant(k_a, k_b, component)

    assert result.offsets.tolist() == [3, -1]
    assert result.num_compared == 6
    assert result.fraction == pytest.approx(5 / 6)
    assert result.component_fraction.tolist() == [0.75, 1.0]
    assert result.residual.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]


def test_agreement_respects_mask_and_breaks_ties_towards_smaller_offset():
    component = np.zeros((1, 4), dtype=int)
    k_b = np.zeros((1, 4), dtype=int)
    k_a = np.array([[2, 2, 1, 1]])
    tie = agreement_modulo_constant(k_a, k_b, component)
    assert tie.offsets.tolist() == [1] and tie.fraction == 0.5

    mask = np.array([[True, True, True, False]])
    masked = agreement_modulo_constant(k_a, k_b, component, mask)
    assert masked.offsets.tolist() == [2] and masked.num_compared == 3


def test_agreement_marks_components_without_compared_pixels():
    component = np.array([[0, 1]])
    result = agreement_modulo_constant(
        np.array([[1, 5]]), np.array([[0, 0]]), component, np.array([[True, False]])
    )
    assert result.offsets.tolist() == [1, 0]
    assert result.component_fraction[0] == 1.0 and math.isnan(result.component_fraction[1])


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ((np.zeros((2, 2), int), np.zeros((2, 3), int), np.zeros((2, 2), int)), "shape mismatch"),
        ((np.zeros((2, 2)), np.zeros((2, 2), int), np.zeros((2, 2), int)), "integer"),
        ((np.zeros((2, 2), int), np.zeros((2, 2), int), -np.ones((2, 2), int)), "no pixels"),
    ],
)
def test_agreement_rejects_bad_input(args, fragment):
    with pytest.raises(ValueError, match=fragment):
        agreement_modulo_constant(*args)


def test_bound_saturation_counts_masked_labels():
    k = np.array([[[-2, 0], [1, 1]], [[-2, -2], [0, 1]]])
    valid = np.array([[True, True], [True, False]])
    sat = bound_saturation(k, -2, 1, valid)
    assert sat.num_labels == 6
    assert sat.at_min == pytest.approx(3 / 6)
    assert sat.at_max == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="outside"):
        bound_saturation(k, -1, 1)


def test_energy_stats_expands_occurrences_and_counts_successes():
    sampleset = dimod.SampleSet.from_samples(
        [{"x": 0}, {"x": 1}], vartype="BINARY", energy=[0.0, 4.0], num_occurrences=[3, 1]
    )
    stats = energy_stats(sampleset, target=0.0)
    assert stats.num_reads == 4
    assert (stats.best, stats.worst, stats.median) == (0.0, 4.0, 0.0)
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(2.0)
    assert stats.success_fraction == pytest.approx(0.75)
    single = energy_stats([2.5])
    assert single.std == 0.0 and single.success_fraction is None


def test_time_to_solution_boundaries():
    assert time_to_solution(2.0, 0.995) == 2.0
    assert time_to_solution(2.0, 0.0) == math.inf
    assert time_to_solution(1.0, 0.5) == pytest.approx(math.log(0.01) / math.log(0.5))
    with pytest.raises(ValueError, match="confidence"):
        time_to_solution(1.0, 0.5, confidence=1.0)
