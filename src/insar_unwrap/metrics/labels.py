"""Gauge-aware comparison of integer label fields.

The unwrapping objective is invariant under ``k -> k + c`` on each connected
component, so two label fields describe the same unwrapped phase whenever they
differ by one constant per component. Comparisons therefore remove the most
frequent per-component difference before counting agreement. This is needed
for free-gauge annealer runs, SNAPHU outputs and product phases alike; aligning
on a single reference pixel is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModuloAgreement:
    """Agreement of two label fields modulo one constant per component.

    Attributes:
        fraction: Fraction of compared pixels with ``k_a - k_b == offset``
            of their component.
        num_compared: Number of compared pixels.
        offsets: Per-component offset, the most frequent ``k_a - k_b`` (ties:
            the smallest value). 0 for components without compared pixels.
        component_fraction: Agreement per component; NaN for components
            without compared pixels.
        residual: ``k_a - k_b - offset`` per pixel (int64, image shape), 0 at
            pixels that were not compared.
    """

    fraction: float
    num_compared: int
    offsets: np.ndarray
    component_fraction: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class BoundSaturation:
    """Share of labels sitting on the ends of the allowed range.

    With a pinned gauge, labels on a bound suggest the range is too narrow.
    With a free gauge the field may sit on one bound by choice of offset;
    labels on both bounds at once are the stronger warning sign.

    Attributes:
        at_min: Fraction of considered labels equal to ``label_min``.
        at_max: Fraction of considered labels equal to ``label_max``.
        num_labels: Number of labels considered.
    """

    at_min: float
    at_max: float
    num_labels: int


def _as_int_array(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array, got dtype {x.dtype}")
    return x.astype(np.int64, copy=False)


def agreement_modulo_constant(
    k_a: np.ndarray,
    k_b: np.ndarray,
    component: np.ndarray,
    mask: np.ndarray | None = None,
) -> ModuloAgreement:
    """Compare two label fields after removing one constant per component.

    Args:
        k_a: Integer label field, e.g. an annealer solution.
        k_b: Integer label field of the same shape, e.g. synthetic truth or
            another solver's output.
        component: Component id per pixel (``-1`` excludes a pixel), usually
            ``UnwrapQubo.component``.
        mask: Optional boolean mask of pixels to compare, e.g. the ``ok`` flags
            from ``UnwrapQubo.decode``. Combined with ``component >= 0``.

    Returns:
        Agreement summary and the per-pixel residual map.

    Raises:
        ValueError: On shape or dtype mismatch, or if no pixel is compared.
    """
    a = _as_int_array(k_a, "k_a")
    b = _as_int_array(k_b, "k_b")
    comp_img = _as_int_array(component, "component")
    if not a.shape == b.shape == comp_img.shape:
        raise ValueError(
            f"shape mismatch: k_a {a.shape}, k_b {b.shape}, component {comp_img.shape}"
        )
    sel = comp_img >= 0
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask must have shape {a.shape}, got {mask.shape}")
        sel &= mask
    if not sel.any():
        raise ValueError("no pixels to compare")

    ncomp = int(comp_img.max()) + 1
    comp = comp_img[sel]
    diff = a[sel] - b[sel]

    # Most frequent difference per component, ties broken by the smallest value.
    dmin = int(diff.min())
    span = int(diff.max()) - dmin + 1
    keys, counts = np.unique(comp * span + (diff - dmin), return_counts=True)
    ucomp, udiff = np.divmod(keys, span)
    order = np.lexsort((udiff, -counts, ucomp))
    first = np.r_[True, ucomp[order][1:] != ucomp[order][:-1]]
    offsets = np.zeros(ncomp, dtype=np.int64)
    offsets[ucomp[order][first]] = udiff[order][first] + dmin

    res = diff - offsets[comp]
    match = res == 0
    total = np.bincount(comp, minlength=ncomp)
    matched = np.bincount(comp, weights=match, minlength=ncomp)
    component_fraction = np.full(ncomp, np.nan)
    np.divide(matched, total, out=component_fraction, where=total > 0)
    residual = np.zeros(a.shape, dtype=np.int64)
    residual[sel] = res
    return ModuloAgreement(
        fraction=float(match.mean()),
        num_compared=int(match.size),
        offsets=offsets,
        component_fraction=component_fraction,
        residual=residual,
    )


def bound_saturation(
    k: np.ndarray,
    label_min: int,
    label_max: int,
    mask: np.ndarray | None = None,
) -> BoundSaturation:
    """Fraction of labels equal to ``label_min`` and to ``label_max``.

    Args:
        k: Integer labels of any shape, e.g. ``(num_samples, rows, cols)``.
        label_min: Smallest allowed label.
        label_max: Largest allowed label.
        mask: Optional boolean mask broadcastable to ``k`` (e.g. ``UnwrapQubo.valid``
            or decode ``ok`` flags); all labels by default.

    Returns:
        Fractions at each bound.

    Raises:
        ValueError: On a non-integer array, an empty selection or labels
            outside ``[label_min, label_max]``.
    """
    k = _as_int_array(k, "k")
    sel = np.ones(k.shape, dtype=bool) if mask is None else np.broadcast_to(mask, k.shape)
    vals = k[sel]
    if vals.size == 0:
        raise ValueError("no labels to evaluate")
    if vals.min() < label_min or vals.max() > label_max:
        raise ValueError(f"labels outside [{label_min}, {label_max}]")
    return BoundSaturation(
        at_min=float(np.mean(vals == label_min)),
        at_max=float(np.mean(vals == label_max)),
        num_labels=int(vals.size),
    )
