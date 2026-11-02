from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "experiment.example.yaml"


@pytest.fixture
def example_dict() -> dict:
    with EXAMPLE_CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(data: dict, name: str = "cfg.yaml") -> Path:
        p = tmp_path / name
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        return p

    return _write


GRID = "/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram"
LAYERS = {
    "wrapped": f"{GRID}/HH/wrappedInterferogram",
    "coherence": f"{GRID}/HH/coherenceMagnitude",
    "mask": f"{GRID}/mask",
}


@pytest.fixture
def make_gunw(tmp_path: Path):
    """Write a tiny GUNW-like HDF5 file (6 x 8, Rev E layout) and return its path.

    Mask code 11 everywhere except: (0, 0) water 111, (0, 1) secondary subswath
    0 (code 10), (0, 2) fill 255, (0, 3) anomaly bit 8 (uint32 only, else 11).
    The interferogram is NaN at (1, 0); coherence is 0.1 at (1, 1), else 0.8.
    A second grid ``shifted`` has the same shape but offset coordinates.
    """
    import h5py
    import numpy as np

    def _make(mask_dtype: str = "uint8", name: str = "gunw.h5") -> Path:
        rows, cols = 6, 8
        r, c = np.mgrid[0:rows, 0:cols]
        psi = 0.9 * c - 0.7 * r
        coh = np.full((rows, cols), 0.8, dtype=np.float32)
        coh[1, 1] = 0.1
        z = (coh * np.exp(1j * psi)).astype(np.complex64)
        z[1, 0] = np.nan
        mask = np.full((rows, cols), 11, dtype=mask_dtype)
        mask[0, 0], mask[0, 1], mask[0, 2] = 111, 10, 255
        if mask_dtype == "uint32":
            mask[0, 3] = 11 | (1 << 8)
        x = 1000.0 + 20.0 * np.arange(cols)
        y = 5000.0 - 20.0 * np.arange(rows)
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            ident = f.create_group("/science/LSAR/identification")
            ident["granuleId"] = b"TEST_GRANULE"
            ident["missionId"] = b"TEST"
            ident["productSpecificationVersion"] = b"1.2.1"
            for group, dx in ((GRID, 0.0), ("/science/LSAR/GUNW/grids/frequencyA/shifted", 20.0)):
                g = f.require_group(group)
                g["xCoordinates"] = x + dx
                g["yCoordinates"] = y
                g["xCoordinateSpacing"] = 20.0
                g["yCoordinateSpacing"] = -20.0
                g["projection"] = np.uint32(32611)
                hh = g.require_group("HH")
                hh["xCoordinates"] = x + dx
                hh["yCoordinates"] = y
                hh["coherenceMagnitude"] = coh
            f[LAYERS["wrapped"]] = z
            m = f.create_dataset(LAYERS["mask"], data=mask)
            m.attrs["_FillValue"] = np.array(255, dtype=mask_dtype)
            m.attrs["description"] = b"three-digit code"
        return path

    return _make
