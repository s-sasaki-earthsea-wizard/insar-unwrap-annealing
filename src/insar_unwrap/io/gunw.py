"""NISAR L2 GUNW access: structure listing and native-grid crops.

Verified layouts (2026-11-02/03):

* product specification 1.2.1 (JPL D-102272 Rev E), sample D00404 generated
  from ALOS-1 PALSAR data: ``mask`` is uint8, a three-digit code
  ``<water><reference subswath><secondary subswath>``, fill 255;
* product specification 1.5.0, PROVISIONAL NISAR product P05023: ``mask`` is
  uint32; bits 0-7 hold the same three-digit code, bits 8-15 and 16-23 are
  anomaly flags of the secondary and reference RSLC, fill 255.

Both are decoded by one rule: a pixel is usable when the mask is not the fill
value, the low byte has non-zero subswath digits (and, by default, water
digit 0), and no anomaly bit 8-23 is set. Bits 24-31 are ignored.

Grids are compared by their coordinate arrays: every layer must sit below a
group whose ``xCoordinates`` / ``yCoordinates`` equal those of the wrapped
layer. Crops are cut on that native grid; nothing is resampled. Optional
complex multilooking (:mod:`insar_unwrap.io.multilook`) averages
non-overlapping blocks of that window after the pixel validity is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from insar_unwrap.io.multilook import block_centres, multilook
from insar_unwrap.phase import wrap

IDENTIFICATION = "/science/LSAR/identification"
_ID_FIELDS = (
    "granuleId",
    "missionId",
    "productType",
    "productVersion",
    "productSpecificationVersion",
    "compositeReleaseId",
    "orbitPassDirection",
    "trackNumber",
    "frameNumber",
    "processingDateTime",
)
_ANOMALY_BITS = 0x00FFFF00


class GunwError(Exception):
    """Raised for missing layers, grid mismatches or invalid crops."""


@dataclass(frozen=True)
class GunwCrop:
    """Wrapped phase, coherence and validity on one window, optionally multilooked.

    All arrays have the crop shape, i.e. the multilooked shape when
    ``looks > 1``. The window covers ``rows * looks`` by ``cols * looks``
    native pixels starting at ``(row0, col0)``.

    Attributes:
        wrapped: Wrapped phase in ``(-pi, pi]`` (float64); NaN where the
            (multilooked) complex value is zero or not finite.
        coherence: Coherence magnitude (float64); block mean over usable
            pixels when multilooked.
        valid: Usable pixels (mask rule, finite non-zero interferogram,
            finite coherence at or above the threshold); when multilooked,
            blocks with enough usable pixels.
        mask_code: Raw mask values when ``looks == 1``; ``None`` when no mask
            layer was read or when multilooked (the codes live on the native
            grid, which must not be mixed with the crop grid).
        row0: First native row of the window.
        col0: First native column of the window.
        x: Easting of the crop columns (pixel or block centres).
        y: Northing of the crop rows.
        epsg: EPSG code of the grid.
        looks: Block size of the multilooking (1: none).
        consistency: ``|sum z| / sum |z|`` per block over usable pixels (1 on
            usable pixels when ``looks == 1``); diagnostic only.
        source: File, granule id, specification version, layer paths and
            window parameters.
    """

    wrapped: np.ndarray
    coherence: np.ndarray
    valid: np.ndarray
    mask_code: np.ndarray | None
    row0: int
    col0: int
    x: np.ndarray
    y: np.ndarray
    epsg: int | None
    looks: int = 1
    consistency: np.ndarray | None = None
    source: dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Crop shape ``(rows, cols)``."""
        return (int(self.wrapped.shape[0]), int(self.wrapped.shape[1]))


_SKIP_ATTRS = {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"}


def _decode(value: object) -> object:
    """Convert HDF5 values to plain Python types for YAML output."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return [v.decode(errors="replace") for v in value.ravel().tolist()]
        if value.dtype.kind in "biuf" and value.size <= 16:
            return value.tolist()
        return f"<array {value.dtype} {list(value.shape)}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _grid_group(ds: h5py.Dataset) -> h5py.Group:
    """Nearest ancestor group that carries xCoordinates and yCoordinates."""
    group = ds.parent
    while True:
        if "xCoordinates" in group and "yCoordinates" in group:
            return group
        if group.name == "/":
            raise GunwError(f"no xCoordinates/yCoordinates above {ds.name}")
        group = group.parent


def _lookup(group: h5py.Group, key: str) -> object:
    """Scalar ``key`` from ``group`` or its nearest ancestor that has it."""
    while True:
        if key in group:
            return _decode(group[key][()])
        if group.name == "/":
            return None
        group = group.parent


def _require(f: h5py.File, path: str) -> h5py.Dataset:
    obj = f.get(path)
    if not isinstance(obj, h5py.Dataset):
        raise GunwError(f"dataset not found: {path}")
    return obj


def identification(f: h5py.File) -> dict[str, object]:
    """Selected scalar fields of ``/science/LSAR/identification``."""
    out: dict[str, object] = {}
    ident = f.get(IDENTIFICATION)
    if isinstance(ident, h5py.Group):
        for key in _ID_FIELDS:
            if key in ident:
                out[key] = _decode(ident[key][()])
    return out


def describe(path: Path | str, root: str = "/science/LSAR/GUNW/grids") -> dict[str, object]:
    """Plain-data description of a GUNW file for ``insar-unwrap inspect``.

    Lists identification fields, every 2-D dataset below ``root`` with shape
    and dtype, every grid (group with coordinates) with spacing, origin, size
    and EPSG code, and the attributes of mask layers.

    Args:
        path: HDF5 file.
        root: Group to walk.

    Returns:
        Nested dict suitable for YAML output.
    """
    layers: dict[str, object] = {}
    grids: dict[str, object] = {}
    masks: dict[str, object] = {}
    with h5py.File(path, "r") as f:
        if root not in f:
            raise GunwError(f"group not found: {root}")

        def visit(name: str, obj: object) -> None:
            full = f"{root}/{name}"
            if isinstance(obj, h5py.Group) and "xCoordinates" in obj and "yCoordinates" in obj:
                x, y = obj["xCoordinates"], obj["yCoordinates"]
                grid: dict[str, object] = {
                    "rows": int(y.shape[0]),
                    "cols": int(x.shape[0]),
                    "x0": float(x[0]),
                    "y0": float(y[0]),
                }
                for key in ("xCoordinateSpacing", "yCoordinateSpacing", "projection"):
                    if key in obj:
                        grid[key] = _decode(obj[key][()])
                grids[full] = grid
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                layers[full] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if name.rsplit("/", 1)[-1] == "mask":
                    masks[full] = {
                        k: _decode(v) for k, v in obj.attrs.items() if k not in _SKIP_ATTRS
                    }

        f[root].visititems(visit)
        return {
            "file": str(path),
            "identification": identification(f),
            "grids": grids,
            "layers": layers,
            "masks": masks,
        }


def usable_mask(code: np.ndarray, fill: int = 255, *, exclude_water: bool = True) -> np.ndarray:
    """Decode GUNW mask values into a usable-pixel flag.

    Args:
        code: Raw mask values (uint8 or uint32).
        fill: Fill value of the layer.
        exclude_water: Treat pixels with water digit 1 as unusable.

    Returns:
        Boolean array, True where the pixel is usable.
    """
    v = np.asarray(code).astype(np.int64)
    low = v & 0xFF
    ok = (v != fill) & ((low // 10) % 10 != 0) & (low % 10 != 0) & ((v & _ANOMALY_BITS) == 0)
    if exclude_water:
        ok &= low // 100 == 0
    return ok


def read_crop(
    path: Path | str,
    *,
    wrapped: str,
    coherence: str,
    mask: str | None,
    row0: int,
    col0: int,
    rows: int,
    cols: int,
    coherence_threshold: float = 0.0,
    exclude_water: bool = True,
    looks: int = 1,
    min_valid_fraction: float = 0.5,
) -> GunwCrop:
    """Read one window of the wrapped interferogram, coherence and mask.

    Pixel validity is decided on the native grid first. With ``looks > 1`` the
    window of ``rows * looks`` by ``cols * looks`` native pixels is then
    complex-multilooked (:func:`insar_unwrap.io.multilook.multilook`).

    Args:
        path: GUNW HDF5 file.
        wrapped: Path of the complex wrapped interferogram.
        coherence: Path of the coherence magnitude on the same grid.
        mask: Path of the mask on the same grid, or ``None`` to skip it.
        row0: First row on the native grid.
        col0: First column on the native grid.
        rows: Crop height (after multilooking).
        cols: Crop width (after multilooking).
        coherence_threshold: Native pixels with lower coherence are not usable.
        exclude_water: Passed to :func:`usable_mask`.
        looks: Square block size of the multilooking; 1 reads native pixels.
        min_valid_fraction: Smallest fraction of usable native pixels for a
            usable block, in ``(0, 1]``. Irrelevant when ``looks == 1``.

    Returns:
        The crop with validity flags and grid metadata.

    Raises:
        GunwError: Missing layers, wrong dtypes, layers on different grids or
            a window outside the grid.
    """
    if rows < 1 or cols < 1 or row0 < 0 or col0 < 0 or looks < 1:
        raise GunwError(
            f"invalid window: row0={row0} col0={col0} rows={rows} cols={cols} looks={looks}"
        )
    if not 0.0 < min_valid_fraction <= 1.0:
        raise GunwError(f"min_valid_fraction must be in (0, 1], got {min_valid_fraction}")
    nrows, ncols = rows * looks, cols * looks
    with h5py.File(path, "r") as f:
        w = _require(f, wrapped)
        c = _require(f, coherence)
        m = _require(f, mask) if mask is not None else None
        if w.dtype.kind != "c":
            raise GunwError(f"{wrapped} must be complex, got {w.dtype}")
        if c.dtype.kind != "f":
            raise GunwError(f"{coherence} must be real floating point, got {c.dtype}")
        if m is not None and m.dtype.kind != "u":
            raise GunwError(f"{mask} must be unsigned integer, got {m.dtype}")

        grid = _grid_group(w)
        gx, gy = grid["xCoordinates"][()], grid["yCoordinates"][()]
        for ds in (c, m):
            if ds is None:
                continue
            other = _grid_group(ds)
            if ds.shape != w.shape or not (
                np.array_equal(other["xCoordinates"][()], gx)
                and np.array_equal(other["yCoordinates"][()], gy)
            ):
                raise GunwError(f"{ds.name} is not on the grid of {wrapped}")
        if w.shape != (gy.size, gx.size):
            raise GunwError(f"{wrapped} shape {w.shape} does not match its coordinates")
        if row0 + nrows > w.shape[0] or col0 + ncols > w.shape[1]:
            raise GunwError(
                f"window exceeds grid {w.shape}: rows {row0}+{nrows}, cols {col0}+{ncols}"
                f" (looks={looks})"
            )

        win = np.s_[row0 : row0 + nrows, col0 : col0 + ncols]
        z = w[win]
        coh = c[win].astype(np.float64)
        code = m[win] if m is not None else None
        fill = int(m.attrs.get("_FillValue", 255)) if m is not None else 255
        projection = _lookup(grid, "projection")
        epsg = int(projection) if projection is not None else None
        source = {
            "file": str(path),
            **identification(f),
            "wrapped": wrapped,
            "coherence": coherence,
            "mask": mask,
            "grid": grid.name,
            "grid_shape": list(w.shape),
            "spacing": [_lookup(grid, k) for k in ("xCoordinateSpacing", "yCoordinateSpacing")],
            "looks": looks,
            "min_valid_fraction": min_valid_fraction if looks > 1 else None,
            "native_window": [row0, col0, nrows, ncols],
        }

    finite = np.isfinite(z) & (z != 0)
    with np.errstate(invalid="ignore"):
        valid = finite & np.isfinite(coh) & (coh >= coherence_threshold)
    if code is not None:
        valid &= usable_mask(code, fill, exclude_water=exclude_water)
    ml = multilook(z, coh, valid, looks, min_valid_fraction=min_valid_fraction)
    zm = ml.interferogram
    nonzero = zm != 0
    psi = np.full(zm.shape, np.nan)
    psi[nonzero] = wrap(np.angle(zm[nonzero]))
    return GunwCrop(
        wrapped=psi,
        coherence=ml.coherence if looks > 1 else coh,
        valid=ml.valid,
        mask_code=code if looks == 1 else None,
        row0=row0,
        col0=col0,
        x=block_centres(gx[col0 : col0 + ncols], looks),
        y=block_centres(gy[row0 : row0 + nrows], looks),
        epsg=epsg,
        looks=looks,
        consistency=ml.consistency,
        source=source,
    )
