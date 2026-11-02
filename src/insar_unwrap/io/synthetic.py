"""Synthetic wrapped interferograms with known integer ambiguities.

These fields exist to check the implementation (sign conventions, encoding,
gauge handling, solver plumbing). They are not a substitute for real-data
evaluation and must be reported separately from it.

The generated "truth" is the noisy unwrapped phase ``phase + noise``: that is
what an ideal unwrapper could recover from the wrapped observation. Whether
the minimum of a given objective coincides with it depends on noise level and
phase gradient (Itoh condition); ``SyntheticField.num_residues`` reports how
often the condition is violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from insar_unwrap.phase import TWO_PI, residues, wrap


@dataclass(frozen=True)
class SyntheticField:
    """A synthetic interferogram and its ground truth.

    Attributes:
        signal: Noise-free unwrapped phase in radians.
        unwrapped: ``signal + noise``; the recoverable truth.
        wrapped: ``wrap(unwrapped)``, in ``(-pi, pi]``.
        k_true: Integer ambiguities with ``unwrapped == wrapped + 2*pi*k_true``.
        noise_std: Standard deviation of the additive phase noise (radians).
        seed: Seed used for the noise draw.
    """

    signal: np.ndarray
    unwrapped: np.ndarray
    wrapped: np.ndarray
    k_true: np.ndarray
    noise_std: float
    seed: int | None

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(rows, cols)``."""
        return self.wrapped.shape  # type: ignore[return-value]

    @property
    def num_residues(self) -> int:
        """Number of non-zero residues; zero means the Itoh condition holds everywhere."""
        return int(np.count_nonzero(residues(self.wrapped)))


def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be (rows, cols) with positive sizes, got {shape}")
    return int(shape[0]), int(shape[1])


def ramp(shape: tuple[int, int], fringes_row: float, fringes_col: float) -> np.ndarray:
    """Planar phase ramp.

    Args:
        shape: ``(rows, cols)``.
        fringes_row: Number of 2*pi cycles accumulated across the row axis.
        fringes_col: Number of 2*pi cycles accumulated across the column axis.

    Returns:
        Unwrapped phase in radians, zero at pixel ``(0, 0)``.
    """
    rows, cols = _check_shape(shape)
    r = np.arange(rows, dtype=float)[:, None] / rows
    c = np.arange(cols, dtype=float)[None, :] / cols
    return TWO_PI * (fringes_row * r + fringes_col * c)


def gaussian_bump(
    shape: tuple[int, int],
    amplitude_cycles: float,
    sigma_px: float,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Isotropic Gaussian phase bump (a crude stand-in for localized deformation).

    Args:
        shape: ``(rows, cols)``.
        amplitude_cycles: Peak height in 2*pi cycles (may be negative).
        sigma_px: Standard deviation in pixels, > 0.
        center: ``(row, col)`` of the peak; defaults to the image centre.

    Returns:
        Unwrapped phase in radians.
    """
    rows, cols = _check_shape(shape)
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    if center is None:
        center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    r = np.arange(rows, dtype=float)[:, None] - center[0]
    c = np.arange(cols, dtype=float)[None, :] - center[1]
    return TWO_PI * amplitude_cycles * np.exp(-(r**2 + c**2) / (2.0 * sigma_px**2))


def synthesize(
    signal: np.ndarray, noise_std: float = 0.0, seed: int | None = None
) -> SyntheticField:
    """Add Gaussian phase noise to an unwrapped signal and wrap it.

    Args:
        signal: Noise-free unwrapped phase, 2-D, finite.
        noise_std: Standard deviation of additive phase noise in radians (>= 0).
        seed: Seed for ``numpy.random.default_rng``. Ignored when ``noise_std == 0``.

    Returns:
        The synthetic field with exact integer ambiguities.

    Raises:
        ValueError: On a non-2-D or non-finite signal or a negative ``noise_std``.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D, got shape {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains non-finite values")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std > 0:
        noise = np.random.default_rng(seed).normal(0.0, noise_std, size=signal.shape)
    else:
        noise = np.zeros_like(signal)
    unwrapped = signal + noise
    wrapped = wrap(unwrapped)
    k_true = np.rint((unwrapped - wrapped) / TWO_PI).astype(np.int64)
    return SyntheticField(
        signal=signal,
        unwrapped=unwrapped,
        wrapped=wrapped,
        k_true=k_true,
        noise_std=float(noise_std),
        seed=seed,
    )
