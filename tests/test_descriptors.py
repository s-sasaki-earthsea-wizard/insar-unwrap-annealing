import numpy as np
import pytest

from insar_unwrap.metrics.descriptors import mean_fringe_rate, tree_label_span
from insar_unwrap.phase import TWO_PI, wrap
from insar_unwrap.qubo import build_unwrap_bqm


def _ramp(shape, f_row, f_col):
    r, c = np.mgrid[0 : shape[0], 0 : shape[1]]
    phi = TWO_PI * (f_row * r + f_col * c)
    return phi, wrap(phi)


def test_mean_fringe_rate_recovers_a_planar_ramp():
    _, psi = _ramp((6, 9), 0.05, 0.3)
    valid = np.ones(psi.shape, bool)
    valid[2, 3] = False
    assert mean_fringe_rate(psi, valid) == pytest.approx(np.hypot(0.05, 0.3))


def test_tree_label_span_matches_the_ideal_labels_without_residues():
    phi, psi = _ramp((4, 8), 0.1, 0.3)
    k_true = np.rint((phi - psi) / TWO_PI).astype(int)
    valid = np.ones(psi.shape, bool)
    valid[:, 4] = False  # two components; the right one spans fewer cycles
    e = build_unwrap_bqm(psi, label_min=-3, label_max=3, valid=valid).edges

    span = tree_label_span(psi.size, e.src, e.dst, e.target, valid)

    left, right = k_true[:, :4], k_true[:, 5:]
    assert span == max(np.ptp(left), np.ptp(right))
    assert tree_label_span(psi.size, e.src[:0], e.dst[:0], e.target[:0], valid) == 0
