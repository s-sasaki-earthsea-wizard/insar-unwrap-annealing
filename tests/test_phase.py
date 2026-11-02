import numpy as np
import pytest

from insar_unwrap.phase import TWO_PI, integer_targets, residues, wrap


def test_wrap_is_half_open_minus_pi_to_pi():
    assert wrap(np.pi) == pytest.approx(np.pi)
    assert wrap(-np.pi) == pytest.approx(np.pi)
    assert wrap(3 * np.pi) == pytest.approx(np.pi)
    x = np.linspace(-20, 20, 2001)
    w = wrap(x)
    assert np.all(w > -np.pi) and np.all(w <= np.pi + 1e-12)
    assert np.allclose(wrap(w), w)
    cycles = (x - w) / TWO_PI
    assert np.allclose(cycles, np.rint(cycles))


def test_integer_targets_match_true_ambiguity_gradient_under_itoh():
    rows, cols = 6, 7
    r, c = np.mgrid[0:rows, 0:cols]
    phi = 0.9 * r - 1.3 * c + 0.2  # |gradient| < pi
    psi = wrap(phi)
    k = np.rint((phi - psi) / TWO_PI).astype(int)
    a_h, a_v = integer_targets(psi)
    assert a_h.shape == (rows, cols - 1) and a_v.shape == (rows - 1, cols)
    assert np.array_equal(a_h, k[:, 1:] - k[:, :-1])
    assert np.array_equal(a_v, k[1:, :] - k[:-1, :])
    assert set(np.unique(np.concatenate([a_h.ravel(), a_v.ravel()]))) <= {-1, 0, 1}


def test_residues_zero_for_smooth_field_and_nonzero_at_vortex():
    r, c = np.mgrid[0:5, 0:5]
    assert not residues(wrap(0.5 * r + 0.4 * c)).any()
    vortex = np.angle((c - 2.5) + 1j * (r - 2.5))  # phase singularity between pixels
    res = residues(vortex)
    assert np.count_nonzero(res) == 1
    assert abs(int(res.sum())) == 1


def test_integer_targets_reject_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        integer_targets(np.zeros(4))
    bad = np.zeros((3, 3))
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        integer_targets(bad)
