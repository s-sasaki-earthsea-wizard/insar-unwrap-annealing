import numpy as np
import pytest

from insar_unwrap.io.gunw import GunwError, read_crop
from insar_unwrap.io.multilook import multilook
from insar_unwrap.phase import wrap

from .conftest import LAYERS


def test_multilook_averages_complex_values_over_usable_pixels_only():
    z = np.array(
        [
            [1.0, 1j, 5.0, 5.0],
            [-1.0, -1j, 5.0, np.nan],
        ],
        dtype=np.complex64,
    )
    coh = np.array([[0.2, 0.4, 0.9, 0.9], [0.6, 0.8, 0.5, np.nan]])
    valid = np.array([[True, True, True, False], [False, False, True, False]])

    ml = multilook(z, coh, valid, 2, min_valid_fraction=0.5)

    # Left block: 1 + 1j over two usable pixels; amplitude sum 2.
    assert ml.interferogram[0, 0] == pytest.approx((1 + 1j) / 2)
    assert ml.coherence[0, 0] == pytest.approx(0.3)
    assert ml.consistency[0, 0] == pytest.approx(np.sqrt(2) / 2)
    assert ml.valid_fraction.tolist() == [[0.5, 0.5]]
    assert ml.valid.tolist() == [[True, True]]
    # The NaN at an unusable pixel does not leak into the block mean.
    assert ml.interferogram[0, 1] == pytest.approx(5.0)
    assert ml.consistency[0, 1] == pytest.approx(1.0)

    strict = multilook(z, coh, valid, 2, min_valid_fraction=0.75)
    assert not strict.valid.any()


def test_multilook_rejects_cancelling_blocks_and_bad_arguments():
    z = np.array([[1.0, -1.0]], dtype=complex)
    ml = multilook(z, np.ones((1, 2)), np.ones((1, 2), bool), 1)
    assert ml.valid.tolist() == [[True, True]]
    assert ml.consistency.tolist() == [[1.0, 1.0]]

    zz = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=complex)
    assert not multilook(zz, np.ones((2, 2)), np.ones((2, 2), bool), 2).valid.any()

    with pytest.raises(ValueError, match="divisible"):
        multilook(np.ones((3, 4), complex), np.ones((3, 4)), np.ones((3, 4), bool), 2)
    with pytest.raises(ValueError, match="min_valid_fraction"):
        multilook(zz, np.ones((2, 2)), np.ones((2, 2), bool), 2, min_valid_fraction=0.0)


def test_read_crop_with_looks_multilooks_the_native_window(make_gunw):
    path = make_gunw("uint32")
    crop = read_crop(
        path, **LAYERS, row0=0, col0=0, rows=3, cols=4, coherence_threshold=0.3, looks=2
    )

    assert crop.shape == (3, 4) and crop.looks == 2
    assert crop.source["native_window"] == [0, 0, 6, 8]
    assert crop.mask_code is None
    assert crop.x.tolist() == [1010.0, 1050.0, 1090.0, 1130.0]
    assert crop.y.tolist() == [4990.0, 4950.0, 4910.0]
    # Block (0, 0): water, bad subswath, NaN and low coherence -> nothing usable.
    # Block (0, 1): fill and anomaly bit leave (1, 2), (1, 3) -> exactly 0.5.
    assert crop.valid[0].tolist() == [False, True, True, True]
    assert crop.valid[1:].all()
    assert crop.wrapped[0, 1] == pytest.approx(float(wrap(0.9 * 2.5 - 0.7 * 1)))
    assert crop.coherence[0, 1] == pytest.approx(0.8)

    strict = read_crop(
        path,
        **LAYERS,
        row0=0,
        col0=0,
        rows=3,
        cols=4,
        coherence_threshold=0.3,
        looks=2,
        min_valid_fraction=0.75,
    )
    assert not strict.valid[0, 1]

    with pytest.raises(GunwError, match="exceeds grid"):
        read_crop(path, **LAYERS, row0=0, col0=0, rows=4, cols=4, looks=2)
