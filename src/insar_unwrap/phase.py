"""Phase conventions shared by the synthetic generator and the QUBO builder.

Conventions (fixed here, tested in ``tests/test_phase.py``):

* Wrapped phase lies in ``(-pi, pi]``.
* The unwrapped phase is ``phi = psi + 2*pi*k`` with integer ambiguity ``k``.
* Edges point "forward": a horizontal edge goes from ``(r, c)`` to ``(r, c+1)``,
  a vertical edge from ``(r, c)`` to ``(r+1, c)``. Arrays of horizontal edges
  have shape ``(rows, cols-1)``; vertical ones ``(rows-1, cols)``.
* For an edge ``s -> t`` the integer target is
  ``a = (W(psi_t - psi_s) - (psi_t - psi_s)) / (2*pi)``, which lies in
  ``{-1, 0, 1}``. Under the Itoh condition (``|phi_t - phi_s| < pi``) the true
  ambiguities satisfy ``k_t - k_s == a`` exactly.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(x: np.ndarray | float) -> np.ndarray:
    """Wrap angles to the half-open interval ``(-pi, pi]``.

    Args:
        x: Angles in radians.

    Returns:
        Wrapped angles with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    return x - TWO_PI * np.ceil((x - np.pi) / TWO_PI)


def _forward_differences(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return psi[:, 1:] - psi[:, :-1], psi[1:, :] - psi[:-1, :]


def integer_targets(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer gradient targets ``a`` for horizontal and vertical edges.

    Args:
        psi: Wrapped phase, 2-D, finite, in ``(-pi, pi]``.

    Returns:
        ``(a_h, a_v)`` int64 arrays of shapes ``(rows, cols-1)`` and
        ``(rows-1, cols)``.

    Raises:
        ValueError: If ``psi`` is not a finite 2-D array.
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 2:
        raise ValueError(f"wrapped phase must be 2-D, got shape {psi.shape}")
    if not np.all(np.isfinite(psi)):
        raise ValueError("wrapped phase contains non-finite values")
    out = []
    for d in _forward_differences(psi):
        a = (wrap(d) - d) / TWO_PI
        out.append(np.rint(a).astype(np.int64))
    return out[0], out[1]


def residues(psi: np.ndarray) -> np.ndarray:
    """Phase residues on every elementary 2x2 loop.

    The loop ``(r,c) -> (r,c+1) -> (r+1,c+1) -> (r+1,c) -> (r,c)`` is summed with
    wrapped differences. A non-zero value marks a point where the Itoh
    condition fails somewhere on the loop; no integer field ``k`` can then make
    every edge residual zero.

    Args:
        psi: Wrapped phase, 2-D.

    Returns:
        int64 array of shape ``(rows-1, cols-1)`` with values in ``{-1, 0, 1}``.
    """
    a_h, a_v = integer_targets(psi)
    # The raw differences around a loop cancel, so the loop sum of wrapped
    # differences equals 2*pi times the loop sum of the integer targets.
    return a_h[:-1, :] + a_v[:, 1:] - a_h[1:, :] - a_v[:, :-1]
