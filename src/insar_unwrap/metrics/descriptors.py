"""Descriptors of a problem instance, recorded next to solver results.

They describe the input, not a solution. Performance should be read against
them (fringe density, residues, label span) rather than against settings such
as the multilook factor, which change several of them at once.

* :func:`mean_fringe_rate`: phase gradient of the whole crop from the angle of
  summed neighbour products. It is the planar part only; a localized
  deformation pattern has steeper fringes than this mean suggests.
* :func:`tree_label_span`: integrate the integer targets ``a`` along a
  breadth-first spanning forest of the edge graph and take the largest label
  range of any component. Without residues this is the exact span of the
  ideal labels. With residues it depends on the tree and is only a rough
  guide for choosing ``label_min`` / ``label_max``; it is not a solution.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from insar_unwrap.phase import TWO_PI


def mean_fringe_rate(wrapped: np.ndarray, valid: np.ndarray) -> float:
    """Mean phase gradient magnitude in cycles per pixel.

    Args:
        wrapped: Wrapped phase, 2-D; values at unusable pixels are ignored.
        valid: Usable pixels.

    Returns:
        ``hypot(g_row, g_col) / (2*pi)`` where each ``g`` is the angle of the
        sum of ``exp(i * (psi_next - psi))`` over pairs of usable neighbours;
        0 if there is no such pair along an axis.
    """
    ok = np.asarray(valid, dtype=bool)
    z = np.where(ok, np.exp(1j * np.where(ok, wrapped, 0.0)), 0)
    g_col = np.angle(np.sum(z[:, 1:] * np.conj(z[:, :-1])))
    g_row = np.angle(np.sum(z[1:, :] * np.conj(z[:-1, :])))
    return float(np.hypot(g_row, g_col) / TWO_PI)


def tree_label_span(
    npix: int, src: np.ndarray, dst: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> int:
    """Largest label range of tree-integrated integer targets over components.

    Args:
        npix: Number of pixels (flat).
        src: Flat source index per edge.
        dst: Flat destination index per edge.
        target: Integer target ``a`` per edge (``k_dst - k_src = a`` ideally).
        valid: Usable pixels (any shape with ``npix`` elements).

    Returns:
        ``max over components of (max k - min k)``; 0 without edges.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    a = np.asarray(target, dtype=np.int64)
    ends = np.concatenate([src, dst])
    others = np.concatenate([dst, src])
    step = np.concatenate([a, -a])
    order = np.argsort(ends, kind="stable")
    nbr, inc = others[order].tolist(), step[order].tolist()
    indptr = np.concatenate([[0], np.cumsum(np.bincount(ends, minlength=npix))]).tolist()
    k = [0] * npix
    seen = [False] * npix
    span = 0
    for p in np.flatnonzero(np.asarray(valid).ravel()).tolist():
        if seen[p]:
            continue
        seen[p] = True
        lo = hi = 0
        queue = deque([p])
        while queue:
            q = queue.popleft()
            for i in range(indptr[q], indptr[q + 1]):
                n = nbr[i]
                if not seen[n]:
                    seen[n] = True
                    k[n] = k[q] + inc[i]
                    lo, hi = min(lo, k[n]), max(hi, k[n])
                    queue.append(n)
        span = max(span, hi - lo)
    return span
