import numpy as np
import pytest

from insar_unwrap.io.synthetic import gaussian_bump, ramp, synthesize
from insar_unwrap.phase import TWO_PI


def test_truth_is_exact_and_wrapped_is_in_range():
    f = synthesize(ramp((16, 20), 2.5, -1.5), noise_std=0.3, seed=1)
    assert f.shape == (16, 20)
    assert f.k_true.dtype == np.int64
    assert np.allclose(f.wrapped + TWO_PI * f.k_true, f.unwrapped)
    assert np.all(f.wrapped > -np.pi) and np.all(f.wrapped <= np.pi)


def test_noise_is_reproducible_by_seed():
    sig = gaussian_bump((12, 12), amplitude_cycles=2.0, sigma_px=3.0)
    a = synthesize(sig, noise_std=0.5, seed=7)
    b = synthesize(sig, noise_std=0.5, seed=7)
    c = synthesize(sig, noise_std=0.5, seed=8)
    assert np.array_equal(a.wrapped, b.wrapped)
    assert not np.array_equal(a.wrapped, c.wrapped)


def test_noise_free_smooth_field_has_no_residues_and_strong_noise_does():
    sig = ramp((32, 32), 3.0, 2.0)
    assert synthesize(sig).num_residues == 0
    assert synthesize(sig, noise_std=1.5, seed=0).num_residues > 0


def test_ramp_starts_at_zero_and_spans_requested_fringes():
    sig = ramp((10, 10), 0.0, 2.0)
    assert sig[0, 0] == 0.0
    assert sig[0, -1] == pytest.approx(TWO_PI * 2.0 * 9 / 10)


@pytest.mark.parametrize(
    "call",
    [
        lambda: ramp((0, 4), 1.0, 1.0),
        lambda: gaussian_bump((4, 4), 1.0, 0.0),
        lambda: synthesize(np.zeros(5)),
        lambda: synthesize(np.full((3, 3), np.inf)),
        lambda: synthesize(np.zeros((3, 3)), noise_std=-1.0),
    ],
)
def test_invalid_arguments_raise(call):
    with pytest.raises(ValueError):
        call()
