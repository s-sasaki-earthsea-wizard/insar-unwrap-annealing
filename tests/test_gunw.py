import numpy as np
import pytest

from insar_unwrap.io.gunw import GunwError, describe, read_crop, usable_mask
from insar_unwrap.phase import wrap

from .conftest import GRID, LAYERS


def test_usable_mask_decodes_three_digit_code_and_anomaly_bits():
    codes = np.array([11, 111, 10, 1, 255, 11 | 1 << 8, 11 | 1 << 16, 11 | 1 << 24], np.uint32)
    assert usable_mask(codes).tolist() == [True, False, False, False, False, False, False, True]
    assert usable_mask(codes, exclude_water=False)[1]


@pytest.mark.parametrize("mask_dtype", ["uint8", "uint32"])
def test_read_crop_returns_window_values_and_validity(make_gunw, mask_dtype):
    path = make_gunw(mask_dtype)
    crop = read_crop(path, **LAYERS, row0=0, col0=0, rows=3, cols=5, coherence_threshold=0.3)

    assert crop.shape == (3, 5)
    r, c = np.mgrid[0:3, 0:5]
    expected_psi = wrap(0.9 * c - 0.7 * r)
    ok = np.isfinite(crop.wrapped)
    assert np.allclose(crop.wrapped[ok], expected_psi[ok], atol=1e-6)
    assert not ok[1, 0]

    expected = np.ones((3, 5), dtype=bool)
    expected[0, :3] = False  # water, invalid subswath, fill
    expected[1, 0] = False  # NaN interferogram
    expected[1, 1] = False  # coherence below threshold
    if mask_dtype == "uint32":
        expected[0, 3] = False  # anomaly bit
    assert np.array_equal(crop.valid, expected)
    assert crop.x.tolist() == [1000.0, 1020.0, 1040.0, 1060.0, 1080.0]
    assert crop.epsg == 32611
    assert crop.source["productSpecificationVersion"] == "1.2.1"


def test_read_crop_rejects_layer_on_another_grid(make_gunw):
    path = make_gunw()
    shifted = "/science/LSAR/GUNW/grids/frequencyA/shifted/HH/coherenceMagnitude"
    with pytest.raises(GunwError, match="not on the grid"):
        read_crop(path, **(LAYERS | {"coherence": shifted}), row0=0, col0=0, rows=2, cols=2)


@pytest.mark.parametrize(
    ("window", "fragment"),
    [((4, 0, 3, 3), "exceeds grid"), ((0, 0, 0, 3), "invalid window")],
)
def test_read_crop_rejects_bad_windows(make_gunw, window, fragment):
    row0, col0, rows, cols = window
    with pytest.raises(GunwError, match=fragment):
        read_crop(make_gunw(), **LAYERS, row0=row0, col0=col0, rows=rows, cols=cols)


def test_describe_lists_grids_layers_and_masks(make_gunw):
    info = describe(make_gunw("uint32"))
    assert info["identification"]["granuleId"] == "TEST_GRANULE"
    assert info["grids"][GRID]["rows"] == 6 and info["grids"][GRID]["projection"] == 32611
    assert info["layers"][LAYERS["wrapped"]] == {"shape": [6, 8], "dtype": "complex64"}
    assert info["masks"][LAYERS["mask"]]["_FillValue"] == 255
