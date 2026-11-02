"""Edge weights of the unwrapping BQM (ADR-001 in the project wiki).

* ``uniform``: weight 1 on every edge between two usable pixels. Control arm.
* ``coherence``: ``w_ij = gamma_i * gamma_j``. Equal to ``gamma**2`` in a
  uniform region, lower when either end is noisy, bounded in ``[0, 1]`` and
  free of extra parameters. Below the coherence threshold pixels are already
  excluded, so with threshold ``tau`` the raw weights lie in ``[tau**2, 1]``.

A weight is the price of putting a discontinuity (``k_j - k_i != a_ij``) on
that edge. Weights do not move residues; they only change where the cuts
between residues are cheapest.

Raw weights are then divided by a scale. ``median`` (default) uses the median
raw weight over the edges that enter the BQM, so the typical edge weighs 1 in
both arms and a fixed SQA ``beta`` means the same in both. ``none`` keeps the
raw weights. Rescaling never moves the optimum, only the energy units seen by
the annealers; the scale is recorded with the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EdgeWeighting = Literal["uniform", "coherence"]
WeightNormalization = Literal["median", "none"]
_COHERENCE_TOL = 1e-6


@dataclass(frozen=True)
class EdgeWeights:
    """Horizontal and vertical edge weights with their provenance.

    Attributes:
        horizontal: Weights of edges ``(r, c) -> (r, c+1)``, shape
            ``(rows, cols-1)``; 0 where an end is unusable.
        vertical: Weights of edges ``(r, c) -> (r+1, c)``, shape
            ``(rows-1, cols)``; 0 where an end is unusable.
        kind: ``"uniform"`` or ``"coherence"``.
        normalization: ``"median"`` or ``"none"``.
        scale: Raw weights were divided by this value.
        raw: Raw weights of the edges with positive weight (1-D), for records.
    """

    horizontal: np.ndarray
    vertical: np.ndarray
    kind: EdgeWeighting
    normalization: WeightNormalization
    scale: float
    raw: np.ndarray

    def summary(self) -> dict[str, object]:
        """Plain-data description for logs and result records."""
        out: dict[str, object] = {
            "kind": self.kind,
            "normalization": self.normalization,
            "scale": round(self.scale, 6),
            "edges": int(self.raw.size),
        }
        if self.raw.size:
            norm = self.raw / self.scale
            pct = np.percentile(self.raw, [5, 50, 95])
            out["raw_p5_p50_p95"] = [round(float(v), 4) for v in pct]
            out["normalized_min_max"] = [round(float(norm.min()), 4), round(float(norm.max()), 4)]
        return out


def edge_weights(
    kind: EdgeWeighting,
    valid: np.ndarray,
    coherence: np.ndarray | None = None,
    *,
    normalization: WeightNormalization = "median",
) -> EdgeWeights:
    """Edge weights for :func:`insar_unwrap.qubo.build.build_unwrap_bqm`.

    Args:
        kind: ``"uniform"`` or ``"coherence"``.
        valid: Usable pixels.
        coherence: Coherence magnitude with the shape of ``valid``; required
            for ``"coherence"``. Values at unusable pixels are ignored.
        normalization: ``"median"`` or ``"none"``.

    Returns:
        Weights for horizontal and vertical edges, 0 where an end is unusable.

    Raises:
        ValueError: Unknown kind or normalization, missing or mismatched
            coherence, or coherence outside ``[0, 1]`` at a usable pixel.
    """
    ok = np.asarray(valid, dtype=bool)
    if ok.ndim != 2:
        raise ValueError(f"valid must be 2-D, got shape {ok.shape}")
    if normalization not in ("median", "none"):
        raise ValueError(f"unknown normalization {normalization!r}")
    if kind == "uniform":
        g = ok.astype(float)
    elif kind == "coherence":
        if coherence is None:
            raise ValueError("coherence weighting needs a coherence array")
        coh = np.asarray(coherence, dtype=float)
        if coh.shape != ok.shape:
            raise ValueError(f"coherence must have shape {ok.shape}, got {coh.shape}")
        vals = coh[ok]
        if (
            not np.all(np.isfinite(vals))
            or np.any(vals < -_COHERENCE_TOL)
            or np.any(vals > 1.0 + _COHERENCE_TOL)
        ):
            raise ValueError("coherence must be finite and in [0, 1] at usable pixels")
        g = np.where(ok, np.clip(coh, 0.0, 1.0), 0.0)
    else:
        raise ValueError(f"unknown edge weighting {kind!r}")

    w_h = g[:, :-1] * g[:, 1:]
    w_v = g[:-1, :] * g[1:, :]
    raw = np.concatenate([w_h.ravel(), w_v.ravel()])
    raw = raw[raw > 0]
    scale = float(np.median(raw)) if normalization == "median" and raw.size else 1.0
    return EdgeWeights(
        horizontal=w_h / scale,
        vertical=w_v / scale,
        kind=kind,
        normalization=normalization,
        scale=scale,
        raw=raw,
    )
