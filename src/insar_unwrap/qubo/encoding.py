"""Encodings of a bounded integer label into binary variables.

Binary ("log") encoding
    ``k = label_min + sum_b c_b x_b`` with ``B = ceil(log2(D))`` bits for
    ``D = label_max - label_min + 1`` labels. The coefficients are
    ``1, 2, ..., 2**(B-2)`` followed by ``c_last = (D-1) - (2**(B-1) - 1)``, so
    every bit string decodes to a label inside ``[label_min, label_max]`` and no
    penalty term is needed. When ``D`` is a power of two this is the ordinary
    binary expansion; otherwise some labels have two bit strings (harmless for
    the objective, but it adds degeneracy).

One-hot encoding
    One bit per label, ``x_l = 1`` iff ``k = label_min + l``. Bit strings with
    zero or several active bits are infeasible; the BQM adds a penalty for them
    (see ``insar_unwrap.qubo.build``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

EncodingKind = Literal["binary", "one_hot"]


@dataclass(frozen=True)
class LabelEncoding:
    """Mapping between integer labels and per-pixel bit vectors.

    Attributes:
        kind: ``"binary"`` or ``"one_hot"``.
        label_min: Smallest representable label (inclusive).
        label_max: Largest representable label (inclusive).
    """

    kind: EncodingKind
    label_min: int
    label_max: int

    def __post_init__(self) -> None:
        if self.kind not in ("binary", "one_hot"):
            raise ValueError(f"unknown encoding {self.kind!r}")
        if self.label_min >= self.label_max:
            raise ValueError("label_min must be strictly less than label_max")

    @property
    def num_labels(self) -> int:
        """Number of representable labels ``D``."""
        return self.label_max - self.label_min + 1

    @property
    def bits_per_pixel(self) -> int:
        """Binary variables per pixel."""
        if self.kind == "one_hot":
            return self.num_labels
        return math.ceil(math.log2(self.num_labels))

    @property
    def coefficients(self) -> np.ndarray:
        """Label offset carried by each bit (float array of length ``bits_per_pixel``).

        For binary encoding the label offset is ``sum_b coefficients[b] * x_b``;
        for one-hot encoding bit ``l`` stands for offset ``l``.
        """
        if self.kind == "one_hot":
            return np.arange(self.num_labels, dtype=float)
        nbits = self.bits_per_pixel
        lower = [2.0**b for b in range(nbits - 1)]
        last = (self.num_labels - 1) - (2 ** (nbits - 1) - 1)
        return np.array(lower + [float(last)])

    def contains(self, k: np.ndarray | int) -> np.ndarray:
        """Element-wise test ``label_min <= k <= label_max``."""
        k = np.asarray(k)
        return (k >= self.label_min) & (k <= self.label_max)

    def bits_for(self, k: int) -> np.ndarray:
        """Canonical bit vector for one label.

        Args:
            k: Label inside ``[label_min, label_max]``.

        Returns:
            int8 array of length ``bits_per_pixel``.

        Raises:
            ValueError: If ``k`` is out of range.
        """
        if not self.label_min <= k <= self.label_max:
            raise ValueError(f"label {k} outside [{self.label_min}, {self.label_max}]")
        u = int(k) - self.label_min
        bits = np.zeros(self.bits_per_pixel, dtype=np.int8)
        if self.kind == "one_hot":
            bits[u] = 1
            return bits
        nbits = self.bits_per_pixel
        lower_max = 2 ** (nbits - 1) - 1
        if u > lower_max:
            bits[-1] = 1
            u -= int(self.coefficients[-1])
        for b in range(nbits - 1):
            bits[b] = (u >> b) & 1
        return bits

    def decode_bits(self, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode bit vectors (last axis) into labels.

        Args:
            bits: Array of 0/1 values with last axis of length ``bits_per_pixel``.

        Returns:
            ``(k, ok)``: int64 labels and a boolean feasibility mask, both with
            the leading shape of ``bits``. Binary encoding is always feasible.
            For one-hot, infeasible entries get ``k = label_min`` and ``ok = False``.
        """
        bits = np.asarray(bits)
        if bits.shape[-1] != self.bits_per_pixel:
            raise ValueError(
                f"expected last axis of length {self.bits_per_pixel}, got {bits.shape[-1]}"
            )
        if self.kind == "binary":
            u = np.rint(bits.astype(float) @ self.coefficients).astype(np.int64)
            return self.label_min + u, np.ones(u.shape, dtype=bool)
        ok = bits.sum(axis=-1) == 1
        u = np.argmax(bits, axis=-1).astype(np.int64)
        return np.where(ok, self.label_min + u, self.label_min), ok
