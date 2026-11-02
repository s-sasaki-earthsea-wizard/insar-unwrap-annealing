import itertools

import dimod
import numpy as np
import pytest

from insar_unwrap.io.synthetic import ramp, synthesize
from insar_unwrap.metrics import agreement_modulo_constant
from insar_unwrap.phase import TWO_PI, wrap
from insar_unwrap.qubo import LabelEncoding, build_unwrap_bqm


def naive_energy(k, psi, valid=None, weight_h=None, weight_v=None):
    """Independent oracle written from the radian definition of the residual.

    residual = (psi_t + 2 pi k_t) - (psi_s + 2 pi k_s) - W(psi_t - psi_s), in cycles.
    """
    rows, cols = psi.shape
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            for dr, dc, wts in ((0, 1, weight_h), (1, 0, weight_v)):
                r2, c2 = r + dr, c + dc
                if r2 >= rows or c2 >= cols:
                    continue
                if valid is not None and not (valid[r, c] and valid[r2, c2]):
                    continue
                w = 1.0 if wts is None else wts[r, c]
                d = psi[r2, c2] - psi[r, c]
                res = d + TWO_PI * (k[r2, c2] - k[r, c]) - wrap(d)
                total += w * (res / TWO_PI) ** 2
    return total


def sample_from_labels(qubo, k):
    """Bit assignment of a label field for the free variables of ``qubo.bqm``."""
    enc = qubo.encoding
    sample = {}
    for r, c in zip(*np.nonzero(qubo.valid), strict=True):
        bits = enc.bits_for(int(k[r, c]))
        for b, v in enumerate(bits):
            lab = qubo.variable_label(r, c, b)
            if lab in qubo.bqm.variables:
                sample[lab] = int(v)
    return sample


@pytest.mark.parametrize("num_labels", range(2, 10))
def test_binary_encoding_covers_exactly_the_label_range(num_labels):
    enc = LabelEncoding("binary", -1, num_labels - 2)
    allbits = np.array(list(itertools.product([0, 1], repeat=enc.bits_per_pixel)))
    k, ok = enc.decode_bits(allbits)
    assert ok.all()
    assert set(k.tolist()) == set(range(enc.label_min, enc.label_max + 1))
    for label in range(enc.label_min, enc.label_max + 1):
        assert enc.decode_bits(enc.bits_for(label))[0] == label


def test_one_hot_decode_flags_infeasible_bits():
    enc = LabelEncoding("one_hot", -2, 1)
    k, ok = enc.decode_bits(np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]))
    assert ok.tolist() == [True, False, False]
    assert k[0] == 0


@pytest.mark.parametrize("encoding", ["binary", "one_hot"])
def test_bqm_energy_equals_independent_objective(encoding):
    rng = np.random.default_rng(3)
    rows, cols = 5, 6
    psi = wrap(rng.uniform(-np.pi, np.pi, (rows, cols)))  # dense residues
    valid = rng.random((rows, cols)) > 0.15
    w_h = rng.uniform(0.0, 2.0, (rows, cols - 1))
    w_v = rng.uniform(0.0, 2.0, (rows - 1, cols))
    w_h[0, 0] = 0.0  # zero weights drop edges
    qubo = build_unwrap_bqm(
        psi,
        label_min=-2,
        label_max=2,
        encoding=encoding,
        valid=valid,
        weight_h=w_h,
        weight_v=w_v,
        reference_mode="per_component",
        reference_value=1,
    )
    for _ in range(20):
        k = rng.integers(-2, 3, (rows, cols))
        k.reshape(-1)[qubo.references] = 1
        expected = naive_energy(k, psi, valid, w_h, w_v)
        assert qubo.bqm.energy(sample_from_labels(qubo, k)) == pytest.approx(expected)
        assert qubo.integer_energy(k) == pytest.approx(expected)


def test_exact_ground_state_recovers_noise_free_truth():
    field = synthesize(ramp((3, 3), 1.2, -0.9))
    assert field.num_residues == 0
    qubo = build_unwrap_bqm(
        field.wrapped, label_min=-1, label_max=2, reference_mode="per_component"
    )
    assert qubo.references.tolist() == [4]  # centre pixel of a 3x3 grid
    best = dimod.ExactSolver().sample(qubo.bqm).first
    assert best.energy == pytest.approx(0.0)
    k, ok = qubo.decode(best.sample)
    assert ok.all()
    assert np.array_equal(k[0], qubo.gauge_fix(field.k_true))


def test_exact_ground_energy_equals_brute_force_over_integer_labels():
    rng = np.random.default_rng(11)
    psi = wrap(rng.uniform(-np.pi, np.pi, (3, 3)))
    # 3 labels: non-power-of-two
    qubo = build_unwrap_bqm(psi, label_min=-1, label_max=1, reference_mode="per_component")
    free = [p for p in range(9) if p != qubo.references[0]]
    best = np.inf
    for combo in itertools.product(range(-1, 2), repeat=len(free)):
        k = np.zeros(9, dtype=int)
        k[free] = combo
        best = min(best, naive_energy(k.reshape(3, 3), psi))
    exact = dimod.ExactSolver().sample(qubo.bqm).first
    assert exact.energy == pytest.approx(best)
    k, _ = qubo.decode(exact.sample)
    assert naive_energy(k[0], psi) == pytest.approx(best)


def test_one_hot_minimizer_is_feasible_and_matches_binary():
    rng = np.random.default_rng(5)
    psi = wrap(rng.uniform(-np.pi, np.pi, (2, 3)))
    one_hot = build_unwrap_bqm(psi, label_min=-1, label_max=1, encoding="one_hot")
    binary = build_unwrap_bqm(psi, label_min=-1, label_max=1, encoding="binary")
    oh = dimod.ExactSolver().sample(one_hot.bqm).first
    bn = dimod.ExactSolver().sample(binary.bqm).first
    _, ok = one_hot.decode(oh.sample)
    assert ok.all()
    assert oh.energy == pytest.approx(bn.energy)
    assert one_hot.one_hot_penalty is not None and one_hot.one_hot_penalty > 1


def test_mask_splits_graph_into_components_with_one_reference_each():
    psi = wrap(ramp((4, 5), 0.5, 0.5))
    valid = np.ones((4, 5), dtype=bool)
    valid[:, 2] = False
    qubo = build_unwrap_bqm(
        psi, label_min=-1, label_max=2, valid=valid, reference_mode="per_component"
    )
    assert qubo.num_components == 2
    assert qubo.references.size == 2
    assert set(np.unique(qubo.component[valid])) == {0, 1}
    assert (qubo.component[~valid] == -1).all()
    for r in range(4):
        for b in range(qubo.encoding.bits_per_pixel):
            assert qubo.variable_label(r, 2, b) not in qubo.bqm.variables
    k, ok = qubo.decode({v: 0 for v in qubo.bqm.variables})
    assert not ok[0][~valid].any() and ok[0][valid].all()


def test_pixel_reference_mode_pins_requested_pixel():
    # 2x4 keeps ExactSolver at 14 free variables; gradients stay below pi.
    field = synthesize(ramp((2, 4), 0.8, 1.5))
    assert field.num_residues == 0
    qubo = build_unwrap_bqm(
        field.wrapped,
        label_min=-1,
        label_max=2,
        reference_mode="pixel",
        reference_pixel=(0, 0),
        reference_value=-1,
    )
    best = dimod.ExactSolver().sample(qubo.bqm).first
    k, _ = qubo.decode(best.sample)
    assert k[0][0, 0] == -1
    assert np.array_equal(k[0], qubo.gauge_fix(field.k_true))


def test_free_gauge_recovers_noise_free_truth_modulo_a_constant():
    field = synthesize(ramp((3, 3), 1.2, -0.9))
    assert field.num_residues == 0
    qubo = build_unwrap_bqm(field.wrapped, label_min=-1, label_max=2)
    assert qubo.reference_mode == "none" and qubo.references.size == 0
    assert qubo.bqm.num_variables == 9 * qubo.encoding.bits_per_pixel
    best = dimod.ExactSolver().sample(qubo.bqm).first
    assert best.energy == pytest.approx(0.0)
    k, ok = qubo.decode(best.sample)
    agreement = agreement_modulo_constant(k[0], field.k_true, qubo.component, ok[0])
    assert agreement.fraction == 1.0
    with pytest.raises(ValueError, match="pinned reference"):
        qubo.gauge_fix(field.k_true)


def test_free_gauge_ground_energy_equals_brute_force_and_bounds_pinned():
    rng = np.random.default_rng(7)
    psi = wrap(rng.uniform(-np.pi, np.pi, (2, 3)))  # residues present
    free = build_unwrap_bqm(psi, label_min=-2, label_max=1)
    pinned = build_unwrap_bqm(psi, label_min=-2, label_max=1, reference_mode="per_component")
    brute = min(
        naive_energy(np.array(combo).reshape(2, 3), psi)
        for combo in itertools.product(range(-2, 2), repeat=6)
    )
    e_free = dimod.ExactSolver().sample(free.bqm).first.energy
    e_pinned = dimod.ExactSolver().sample(pinned.bqm).first.energy
    assert e_free == pytest.approx(brute)
    assert e_free <= e_pinned + 1e-12


def test_bqm_stays_sparse_on_a_32x32_grid():
    field = synthesize(ramp((32, 32), 1.0, 1.5), noise_std=0.2, seed=0)
    qubo = build_unwrap_bqm(field.wrapped, label_min=-2, label_max=1)
    npix, nedges, nbits = 32 * 32, 2 * 32 * 31, 2
    assert qubo.bqm.num_variables == npix * nbits  # free gauge: no fixed bits
    assert qubo.bqm.num_interactions <= npix * 1 + nedges * nbits**2
    assert qubo.summary()["edges"] == nedges


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"wrapped": np.full((3, 3), 4.0)}, "pass the wrapped"),
        ({"reference_mode": "per_component", "reference_value": 5}, "outside"),
        ({"reference_value": 0}, "pinned reference_mode"),
        ({"reference_mode": "bogus"}, "unknown reference_mode"),
        ({"reference_mode": "pixel"}, "reference_pixel is required"),
        ({"reference_pixel": (0, 0)}, "only allowed"),
        ({"one_hot_penalty": 10.0}, "only meaningful"),
        ({"weight_h": np.ones((2, 2))}, "weight_h must have shape"),
        ({"weight_v": -np.ones((2, 3))}, "non-negative"),
        ({"valid": np.zeros((3, 3), dtype=bool)}, "no valid pixels"),
        ({"label_min": 1, "label_max": 1}, "strictly less"),
    ],
)
def test_invalid_arguments_raise(kwargs, fragment):
    args = {"wrapped": np.zeros((3, 3)), "label_min": -1, "label_max": 1} | kwargs
    with pytest.raises(ValueError, match=fragment):
        build_unwrap_bqm(**args)


def test_pixel_mode_rejects_disconnected_graph():
    valid = np.ones((3, 3), dtype=bool)
    valid[:, 1] = False
    with pytest.raises(ValueError, match="2 components"):
        build_unwrap_bqm(
            np.zeros((3, 3)),
            label_min=-1,
            label_max=1,
            valid=valid,
            reference_mode="pixel",
            reference_pixel=(0, 0),
        )


def test_decode_rejects_incomplete_samples():
    qubo = build_unwrap_bqm(np.zeros((2, 2)), label_min=-1, label_max=1)
    sample = {v: 0 for v in list(qubo.bqm.variables)[1:]}
    with pytest.raises(ValueError, match="missing"):
        qubo.decode(sample)
