"""Complex multilooking of native-grid crops.

Non-overlapping ``looks x looks`` blocks are averaged:

* interferogram: complex mean over usable pixels. GUNW interferograms carry
  amplitude, so the mean is amplitude-weighted; the multilooked wrapped phase
  is its angle. Phases are never averaged directly.
* coherence: mean of the product coherence over usable pixels. This is a
  proxy, not a coherence re-estimated from the multilooked interferogram.
* validity: a block is usable when the fraction of usable pixels is at least
  ``min_valid_fraction`` and the complex sum is non-zero.
* consistency ``|sum z| / sum |z|`` over usable pixels: a diagnostic only. It
  drops with noise and also with the fringe gradient inside the block, so it
  is not used as a weight.

With ``looks == 1`` the blocks are single pixels and the inputs come back
unchanged (as complex128 / float64), with consistency 1 on usable pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Multilooked:
    """Block-averaged interferogram, coherence and validity.

    Attributes:
        interferogram: Complex mean over usable pixels (complex128); 0 where
            a block has no usable pixel.
        coherence: Mean coherence over usable pixels; NaN where none.
        valid: Usable blocks.
        valid_fraction: Fraction of usable pixels per block.
        consistency: ``|sum z| / sum |z|`` over usable pixels; NaN where the
            amplitude sum is zero.
    """

    interferogram: np.ndarray
    coherence: np.ndarray
    valid: np.ndarray
    valid_fraction: np.ndarray
    consistency: np.ndarray


def _block_sum(x: np.ndarray, looks: int) -> np.ndarray:
    rows, cols = x.shape[0] // looks, x.shape[1] // looks
    return x.reshape(rows, looks, cols, looks).sum(axis=(1, 3))


def block_centres(coords: np.ndarray, looks: int) -> np.ndarray:
    """Mean coordinate of each run of ``looks`` consecutive pixel centres."""
    coords = np.asarray(coords, dtype=float)
    if coords.size % looks:
        raise ValueError(f"{coords.size} coordinates are not divisible by looks={looks}")
    return coords.reshape(-1, looks).mean(axis=1)


def multilook(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    valid: np.ndarray,
    looks: int,
    *,
    min_valid_fraction: float = 0.5,
) -> Multilooked:
    """Average a native-grid window over ``looks x looks`` blocks.

    Args:
        interferogram: Complex interferogram, 2-D. Values at unusable pixels
            are ignored (they may be NaN).
        coherence: Coherence magnitude on the same grid. Values at unusable
            pixels are ignored.
        valid: Usable pixels on the native grid.
        looks: Block size in pixels along both axes.
        min_valid_fraction: Smallest fraction of usable pixels for a usable
            block, in ``(0, 1]``.

    Returns:
        The block averages.

    Raises:
        ValueError: On mismatched shapes, a shape not divisible by ``looks``,
            ``looks < 1`` or ``min_valid_fraction`` outside ``(0, 1]``.
    """
    z = np.asarray(interferogram)
    coh = np.asarray(coherence, dtype=float)
    ok = np.asarray(valid, dtype=bool)
    if z.ndim != 2 or z.shape != coh.shape or z.shape != ok.shape:
        raise ValueError(
            f"interferogram, coherence and valid must share one 2-D shape, got "
            f"{z.shape}, {coh.shape}, {ok.shape}"
        )
    if looks < 1:
        raise ValueError(f"looks must be >= 1, got {looks}")
    if z.shape[0] % looks or z.shape[1] % looks:
        raise ValueError(f"shape {z.shape} is not divisible by looks={looks}")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction must be in (0, 1], got {min_valid_fraction}")

    zc = np.where(ok, z, 0).astype(np.complex128)
    count = _block_sum(ok.astype(np.int64), looks)
    zsum = _block_sum(zc, looks)
    amp = _block_sum(np.abs(zc), looks)
    csum = _block_sum(np.where(ok, coh, 0.0), looks)
    fraction = count / float(looks * looks)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_coh = np.where(count > 0, csum / count, np.nan)
        consistency = np.where(amp > 0, np.abs(zsum) / amp, np.nan)
        mean_z = np.where(count > 0, zsum / np.maximum(count, 1), 0)
    usable = (count > 0) & (fraction >= min_valid_fraction) & (zsum != 0)
    return Multilooked(
        interferogram=mean_z,
        coherence=mean_coh,
        valid=usable,
        valid_fraction=fraction,
        consistency=consistency,
    )
