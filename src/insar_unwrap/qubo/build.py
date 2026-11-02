"""Sparse BQM for 2-D phase unwrapping on a 4-neighbour grid.

Objective (integer form, in units of cycles squared)::

    E(k) = sum_{(s -> t) in edges} w_st * (k_t - k_s - a_st)**2

with ``a_st`` from :func:`insar_unwrap.phase.integer_targets`. Multiplying by
``(2*pi)**2`` gives the weighted squared residual between the unwrapped
gradient ``(psi_t + 2*pi*k_t) - (psi_s + 2*pi*k_s)`` and the wrapped gradient
``W(psi_t - psi_s)``. This is the L2 cost of Kelany et al. (2020), eq. (7),
without their unary regularizer.

Design choices made explicit here:

* Invalid pixels carry no variables; edges touching them, and edges with zero
  weight, are dropped.
* Connected components are taken on the remaining edge graph. Each component
  has its own gauge freedom ``k -> k + c``.
* ``none`` (default) leaves the gauge free: every valid pixel keeps all its
  bits, the ground state is degenerate under per-component shifts that stay
  inside the label range, and label fields must be compared modulo one
  constant per component (:mod:`insar_unwrap.metrics`). This avoids the cheap
  bulk-shift trap of a single pinned pixel.
* ``per_component`` pins the member pixel nearest to each component centroid
  (ties: lowest row-major index), so a symmetric label range covers smooth
  fields with the fewest labels. ``pixel`` pins one explicit pixel and
  requires a single component. Both are kept for exact-optimum tests and as a
  comparison arm.
* Reference bits are fixed (removed from the BQM), not penalized.
* Variable labels are ``flat_pixel_index * bits_per_pixel + bit``.
* The BQM is assembled from coefficient arrays whose size is linear in the
  number of edges. No dense matrix is ever formed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import dimod
import numpy as np

from insar_unwrap.phase import integer_targets
from insar_unwrap.qubo.encoding import EncodingKind, LabelEncoding

ReferenceMode = Literal["none", "per_component", "pixel"]
_REFERENCE_MODES = ("none", "per_component", "pixel")

_RANGE_TOL = 1e-9


@dataclass(frozen=True)
class EdgeList:
    """Directed grid edges ``src -> dst`` that survive masking.

    Attributes:
        src: Flat (row-major) index of the source pixel.
        dst: Flat index of the destination pixel (right or down neighbour).
        target: Integer gradient target ``a`` per edge.
        weight: Non-negative weight per edge (all > 0 after filtering).
    """

    src: np.ndarray
    dst: np.ndarray
    target: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return int(self.src.size)


def unwrap_energy(k: np.ndarray, edges: EdgeList) -> float:
    """Integer-form objective ``sum w * (k_dst - k_src - a)**2``.

    Args:
        k: Integer labels with the image shape (or already flat).
        edges: Edge list from :func:`build_unwrap_bqm`.

    Returns:
        The objective value as a float.
    """
    flat = np.asarray(k).reshape(-1)
    r = flat[edges.dst] - flat[edges.src] - edges.target
    return float(np.sum(edges.weight * r.astype(float) ** 2))


@dataclass(frozen=True)
class UnwrapQubo:
    """A phase-unwrapping BQM together with everything needed to interpret it.

    Attributes:
        bqm: Binary quadratic model (BINARY vartype). Reference bits, if any,
            are fixed and absent from it.
        encoding: Label encoding used for every pixel.
        shape: Image shape ``(rows, cols)``.
        valid: Boolean mask of pixels that carry labels.
        component: Component id per pixel (``-1`` for invalid pixels).
        reference_mode: ``"none"``, ``"per_component"`` or ``"pixel"``.
        references: Flat indices of the pinned pixel of each component, ordered
            by component id; empty when ``reference_mode == "none"``.
        reference_value: Label assigned to every reference pixel; ``None``
            when the gauge is free.
        edges: Edges entering the objective.
        one_hot_penalty: Penalty weight for one-hot constraints (``None`` for
            binary encoding).
    """

    bqm: dimod.BinaryQuadraticModel
    encoding: LabelEncoding
    shape: tuple[int, int]
    valid: np.ndarray
    component: np.ndarray
    reference_mode: ReferenceMode
    references: np.ndarray
    reference_value: int | None
    edges: EdgeList
    one_hot_penalty: float | None = None
    _fixed_bits: dict[int, int] = field(default_factory=dict, repr=False)

    @property
    def num_components(self) -> int:
        """Number of connected components of the valid-pixel graph."""
        return int(self.component.max()) + 1

    def variable_label(self, row: int, col: int, bit: int) -> int:
        """BQM label of one bit of one pixel."""
        return (row * self.shape[1] + col) * self.encoding.bits_per_pixel + bit

    def summary(self) -> dict[str, object]:
        """Plain-data description for logs and result records."""
        return {
            "shape": list(self.shape),
            "valid_pixels": int(self.valid.sum()),
            "components": self.num_components,
            "edges": len(self.edges),
            "encoding": self.encoding.kind,
            "label_range": [self.encoding.label_min, self.encoding.label_max],
            "bits_per_pixel": self.encoding.bits_per_pixel,
            "reference_mode": self.reference_mode,
            "reference_value": self.reference_value,
            "num_variables": int(self.bqm.num_variables),
            "num_interactions": int(self.bqm.num_interactions),
            "one_hot_penalty": self.one_hot_penalty,
        }

    def integer_energy(self, k: np.ndarray) -> float:
        """Objective of a label field in integer form (no penalty terms)."""
        return unwrap_energy(k, self.edges)

    def gauge_fix(self, k: np.ndarray) -> np.ndarray:
        """Shift a label field per component so that references equal ``reference_value``.

        Use this to bring an external label field (synthetic truth, another
        solver's output) into the gauge of a pinned BQM for exact-optimum
        checks. Invalid pixels are set to 0. For general comparisons use
        :func:`insar_unwrap.metrics.agreement_modulo_constant`.

        Args:
            k: Integer label field with the image shape.

        Returns:
            int64 array with the image shape.

        Raises:
            ValueError: If the gauge is free (``reference_mode == "none"``).
        """
        if self.reference_mode == "none":
            raise ValueError(
                "gauge_fix needs a pinned reference; with reference_mode 'none' compare "
                "labels with metrics.agreement_modulo_constant"
            )
        k = np.asarray(k).reshape(-1).astype(np.int64)
        comp = self.component.reshape(-1)
        out = np.zeros_like(k)
        sel = comp >= 0
        shift = self.reference_value - k[self.references]
        out[sel] = k[sel] + shift[comp[sel]]
        return out.reshape(self.shape)

    def decode(self, samples: object) -> tuple[np.ndarray, np.ndarray]:
        """Decode samples into label fields.

        Args:
            samples: A ``dimod.SampleSet``, a mapping ``{label: value}`` or any
                other samples-like accepted by ``dimod.as_samples``.

        Returns:
            ``(k, ok)`` with shapes ``(num_samples, rows, cols)``. ``ok`` is
            False for invalid pixels and for pixels whose bits violate the
            one-hot constraint; ``k`` is 0 there.

        Raises:
            ValueError: If a label of a free (non-reference) variable is missing.
        """
        if isinstance(samples, Mapping):
            samples = [samples]
        arr, labels = dimod.as_samples(samples)
        nbits = self.encoding.bits_per_pixel
        npix = self.shape[0] * self.shape[1]
        bits = np.zeros((arr.shape[0], npix * nbits), dtype=np.int8)
        for lab, val in self._fixed_bits.items():
            bits[:, lab] = val
        cols = np.asarray(labels, dtype=np.int64)
        bits[:, cols] = arr
        missing = set(self.bqm.variables) - set(labels)
        if missing:
            raise ValueError(f"{len(missing)} BQM variables missing from samples")
        k, ok = self.encoding.decode_bits(bits.reshape(arr.shape[0], npix, nbits))
        ok &= self.valid.reshape(1, -1)
        k = np.where(ok, k, 0)
        return k.reshape(-1, *self.shape), ok.reshape(-1, *self.shape)


def _grid_edges(
    shape: tuple[int, int],
    a_h: np.ndarray,
    a_v: np.ndarray,
    w_h: np.ndarray,
    w_v: np.ndarray,
    valid: np.ndarray,
) -> EdgeList:
    rows, cols = shape
    idx = np.arange(rows * cols).reshape(shape)
    src = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    dst = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    target = np.concatenate([a_h.ravel(), a_v.ravel()])
    weight = np.concatenate([w_h.ravel(), w_v.ravel()]).astype(float)
    vflat = valid.ravel()
    keep = vflat[src] & vflat[dst] & (weight > 0)
    return EdgeList(src[keep], dst[keep], target[keep], weight[keep])


def _components(npix: int, valid_flat: np.ndarray, edges: EdgeList) -> tuple[np.ndarray, int]:
    """Connected components of the edge graph over valid pixels (row-major ids)."""
    ends = np.concatenate([edges.src, edges.dst])
    others = np.concatenate([edges.dst, edges.src])
    order = np.argsort(ends, kind="stable")
    nbr = others[order].tolist()
    indptr = np.concatenate([[0], np.cumsum(np.bincount(ends, minlength=npix))]).tolist()
    comp = np.full(npix, -1, dtype=np.int64)
    ncomp = 0
    for p in np.flatnonzero(valid_flat).tolist():
        if comp[p] >= 0:
            continue
        comp[p] = ncomp
        stack = [p]
        while stack:
            q = stack.pop()
            for nb in nbr[indptr[q] : indptr[q + 1]]:
                if comp[nb] < 0:
                    comp[nb] = ncomp
                    stack.append(nb)
        ncomp += 1
    return comp, ncomp


def _centroid_references(comp: np.ndarray, ncomp: int, cols: int) -> np.ndarray:
    members = np.flatnonzero(comp >= 0)
    cid = comp[members]
    r, c = np.divmod(members, cols)
    count = np.bincount(cid, minlength=ncomp).astype(float)
    mr = np.bincount(cid, weights=r, minlength=ncomp) / count
    mc = np.bincount(cid, weights=c, minlength=ncomp) / count
    d2 = (r - mr[cid]) ** 2 + (c - mc[cid]) ** 2
    order = np.lexsort((members, d2, cid))
    first = np.r_[True, cid[order][1:] != cid[order][:-1]]
    return members[order][first]


def _check_weights(w: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if w is None:
        return np.ones(shape)
    w = np.asarray(w, dtype=float)
    if w.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return w


def safe_one_hot_penalty(edges: EdgeList, npix: int, num_labels: int) -> float:
    """Penalty weight that makes every one-hot violation strictly suboptimal.

    Every pairwise cost is non-negative, so switching off surplus bits never
    raises the edge cost, and switching on one bit of an empty pixel raises it
    by at most ``sum_{e at p} w_e * max_{l,m} (l - m - a_e)**2``. A penalty
    larger than the maximum of that sum over pixels therefore guarantees that
    all minimizers are feasible.

    Args:
        edges: Edge list of the problem.
        npix: Number of pixels in the image.
        num_labels: Labels per pixel ``D``.

    Returns:
        ``1 + max_p sum_{e at p} w_e * (D - 1 + |a_e|)**2``.
    """
    if len(edges) == 0:
        return 1.0
    worst = edges.weight * (num_labels - 1 + np.abs(edges.target)) ** 2
    per_pixel = np.bincount(edges.src, weights=worst, minlength=npix) + np.bincount(
        edges.dst, weights=worst, minlength=npix
    )
    return 1.0 + float(per_pixel.max())


def build_unwrap_bqm(
    wrapped: np.ndarray,
    *,
    label_min: int,
    label_max: int,
    encoding: EncodingKind = "binary",
    valid: np.ndarray | None = None,
    weight_h: np.ndarray | None = None,
    weight_v: np.ndarray | None = None,
    reference_mode: ReferenceMode = "none",
    reference_pixel: tuple[int, int] | None = None,
    reference_value: int | None = None,
    one_hot_penalty: float | None = None,
) -> UnwrapQubo:
    """Build the phase-unwrapping BQM for one wrapped image.

    Args:
        wrapped: Wrapped phase in ``(-pi, pi]``, 2-D. Values at invalid pixels
            are ignored (they may be NaN).
        label_min: Smallest allowed integer ambiguity (inclusive).
        label_max: Largest allowed integer ambiguity (inclusive).
        encoding: ``"binary"`` or ``"one_hot"``.
        valid: Boolean mask of usable pixels; all pixels by default.
        weight_h: Weights of horizontal edges, shape ``(rows, cols-1)``; 1 by default.
        weight_v: Weights of vertical edges, shape ``(rows-1, cols)``; 1 by default.
        reference_mode: ``"none"`` (free gauge), ``"per_component"`` or
            ``"pixel"``.
        reference_pixel: ``(row, col)``; required iff ``reference_mode == "pixel"``.
        reference_value: Label pinned at every reference pixel; 0 when
            omitted. Only allowed with a pinned ``reference_mode``.
        one_hot_penalty: Penalty weight for one-hot encoding. ``None`` selects
            :func:`safe_one_hot_penalty`. Must be ``None`` for binary encoding.

    Returns:
        The BQM and its interpretation data.

    Raises:
        ValueError: On inconsistent shapes, out-of-range wrapped values,
            invalid reference settings or an unusable penalty.
    """
    enc = LabelEncoding(encoding, label_min, label_max)
    psi = np.asarray(wrapped, dtype=float)
    if psi.ndim != 2 or min(psi.shape) < 1:
        raise ValueError(f"wrapped phase must be a non-empty 2-D array, got shape {psi.shape}")
    shape: tuple[int, int] = (int(psi.shape[0]), int(psi.shape[1]))
    rows, cols = shape
    npix = rows * cols
    if valid is None:
        valid = np.ones(shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid mask must have shape {shape}, got {valid.shape}")
    if not valid.any():
        raise ValueError("no valid pixels")
    vals = psi[valid]
    if not np.all(np.isfinite(vals)):
        raise ValueError("wrapped phase is non-finite at valid pixels")
    if np.any(np.abs(vals) > np.pi + _RANGE_TOL):
        raise ValueError(
            "wrapped phase must lie in (-pi, pi]; pass the wrapped, not unwrapped, phase"
        )
    if reference_mode not in _REFERENCE_MODES:
        raise ValueError(f"unknown reference_mode {reference_mode!r}")
    if reference_pixel is not None and reference_mode != "pixel":
        raise ValueError("reference_pixel is only allowed when reference_mode == 'pixel'")
    if reference_mode == "none":
        if reference_value is not None:
            raise ValueError("reference_value is only allowed with a pinned reference_mode")
    else:
        reference_value = 0 if reference_value is None else int(reference_value)
        if not enc.label_min <= reference_value <= enc.label_max:
            raise ValueError(
                f"reference_value {reference_value} outside [{enc.label_min}, {enc.label_max}]"
            )
    if enc.kind == "binary" and one_hot_penalty is not None:
        raise ValueError("one_hot_penalty is only meaningful with one_hot encoding")
    if one_hot_penalty is not None and not one_hot_penalty > 0:
        raise ValueError("one_hot_penalty must be positive")

    a_h, a_v = integer_targets(np.where(valid, psi, 0.0))
    w_h = _check_weights(weight_h, (rows, cols - 1), "weight_h")
    w_v = _check_weights(weight_v, (rows - 1, cols), "weight_v")
    edges = _grid_edges(shape, a_h, a_v, w_h, w_v, valid)

    valid_flat = valid.ravel()
    comp, ncomp = _components(npix, valid_flat, edges)
    if reference_mode == "none":
        references = np.zeros(0, dtype=np.int64)
    elif reference_mode == "per_component":
        references = _centroid_references(comp, ncomp, cols)
    else:
        if reference_pixel is None:
            raise ValueError("reference_pixel is required when reference_mode == 'pixel'")
        r0, c0 = reference_pixel
        if not (0 <= r0 < rows and 0 <= c0 < cols) or not valid[r0, c0]:
            raise ValueError(f"reference_pixel {reference_pixel} is outside the image or invalid")
        if ncomp != 1:
            raise ValueError(
                f"reference_mode 'pixel' needs a connected valid graph, found {ncomp} "
                "components; use 'per_component'"
            )
        references = np.array([r0 * cols + c0], dtype=np.int64)

    nbits = enc.bits_per_pixel
    pix = np.flatnonzero(valid_flat)
    pos = np.full(npix, -1, dtype=np.int64)
    pos[pix] = np.arange(pix.size)
    labels = (pix[:, None] * nbits + np.arange(nbits)).ravel()
    linear = np.zeros(pix.size * nbits)
    qrow: list[np.ndarray] = []
    qcol: list[np.ndarray] = []
    qval: list[np.ndarray] = []
    offset = 0.0
    coef = enc.coefficients
    s_pos, t_pos = pos[edges.src] * nbits, pos[edges.dst] * nbits
    w, a = edges.weight, edges.target.astype(float)
    bi, bj = np.triu_indices(nbits, k=1)

    penalty: float | None = None
    if enc.kind == "binary":
        # w * (u_t - u_s - a)^2 with u = sum_b c_b x_b and x^2 = x.
        for b in range(nbits):
            np.add.at(linear, t_pos + b, w * (coef[b] ** 2 - 2.0 * a * coef[b]))
            np.add.at(linear, s_pos + b, w * (coef[b] ** 2 + 2.0 * a * coef[b]))
        strength = np.bincount(pos[edges.src], weights=w, minlength=pix.size) + np.bincount(
            pos[edges.dst], weights=w, minlength=pix.size
        )
        base = np.arange(pix.size) * nbits
        for b1, b2 in zip(bi, bj, strict=True):
            qrow.append(base + b1)
            qcol.append(base + b2)
            qval.append(2.0 * coef[b1] * coef[b2] * strength)
        for b1 in range(nbits):
            for b2 in range(nbits):
                qrow.append(t_pos + b1)
                qcol.append(s_pos + b2)
                qval.append(-2.0 * w * coef[b1] * coef[b2])
        offset = float(np.sum(w * a**2))
    else:
        # Assignment form: sum_{l,m} w * (l - m - a)^2 * x_{t,l} * x_{s,m}.
        for lt in range(nbits):
            for ls in range(nbits):
                qrow.append(t_pos + lt)
                qcol.append(s_pos + ls)
                qval.append(w * (lt - ls - a) ** 2)
        penalty = (
            safe_one_hot_penalty(edges, npix, enc.num_labels)
            if one_hot_penalty is None
            else float(one_hot_penalty)
        )
        # penalty * (sum_l x_l - 1)^2 = penalty * (1 - sum_l x_l + 2 sum_{l<l'} x_l x_l')
        linear -= penalty
        base = np.arange(pix.size) * nbits
        for b1, b2 in zip(bi, bj, strict=True):
            qrow.append(base + b1)
            qcol.append(base + b2)
            qval.append(np.full(pix.size, 2.0 * penalty))
        offset = penalty * pix.size

    if qrow:
        row = np.concatenate(qrow)
        col = np.concatenate(qcol)
        val = np.concatenate(qval)
        nz = val != 0
        quadratic = (row[nz], col[nz], val[nz])
    else:
        empty = np.zeros(0, dtype=np.int64)
        quadratic = (empty, empty, np.zeros(0))
    bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(
        linear, quadratic, offset, dimod.BINARY, variable_order=labels.tolist()
    )

    fixed: dict[int, int] = {}
    if reference_value is not None:
        ref_bits = enc.bits_for(reference_value)
        fixed = {int(p) * nbits + b: int(ref_bits[b]) for p in references for b in range(nbits)}
        bqm.fix_variables(fixed)

    return UnwrapQubo(
        bqm=bqm,
        encoding=enc,
        shape=shape,
        valid=valid,
        component=comp.reshape(shape),
        reference_mode=reference_mode,
        references=references.astype(np.int64),
        reference_value=reference_value,
        edges=edges,
        one_hot_penalty=penalty,
        _fixed_bits=fixed,
    )
