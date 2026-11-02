import dimod
import numpy as np
import pytest

from insar_unwrap.phase import residues
from insar_unwrap.qubo import build_unwrap_bqm, edge_weights


def test_coherence_weights_are_products_divided_by_the_median():
    coh = np.array([[0.5, 1.0, 0.8], [0.4, np.nan, 0.6]])
    valid = np.array([[True, True, True], [True, False, True]])

    w = edge_weights("coherence", valid, coh)

    raw = np.array([0.5, 0.8, 0.2, 0.48])  # h: 0.5*1, 1*0.8; v: 0.5*0.4, 0.8*0.6
    assert w.scale == pytest.approx(np.median(raw))
    assert w.horizontal * w.scale == pytest.approx(np.array([[0.5, 0.8], [0.0, 0.0]]))
    assert w.vertical * w.scale == pytest.approx(np.array([[0.2, 0.0, 0.48]]))
    assert w.summary()["edges"] == 4

    none = edge_weights("coherence", valid, coh, normalization="none")
    assert none.scale == 1.0 and none.horizontal[0, 1] == pytest.approx(0.8)

    uniform = edge_weights("uniform", valid)
    assert uniform.scale == 1.0
    assert uniform.horizontal.tolist() == [[1.0, 1.0], [0.0, 0.0]]


@pytest.mark.parametrize(
    ("coherence", "fragment"),
    [(None, "needs a coherence"), (np.full((2, 2), 1.5), "in \\[0, 1\\]"), (np.ones(3), "shape")],
)
def test_coherence_weights_reject_bad_coherence(coherence, fragment):
    with pytest.raises(ValueError, match=fragment):
        edge_weights("coherence", np.ones((2, 2), bool), coherence)


def _cut_edges(psi, weights):
    """Edges with a non-zero residual in every ground state of the 2x2 BQM."""
    q = build_unwrap_bqm(
        psi,
        label_min=-1,
        label_max=0,
        weight_h=weights.horizontal,
        weight_v=weights.vertical,
    )
    sampleset = dimod.ExactSolver().sample(q.bqm)
    ground = sampleset.lowest()
    k, _ = q.decode(ground)
    cuts = set()
    for field in k.reshape(len(k), -1):
        r = field[q.edges.dst] - field[q.edges.src] - q.edges.target
        cuts.add(tuple(int(i) for i in np.flatnonzero(r)))
    return sampleset.first.energy, cuts, q.edges


def test_coherence_weights_put_the_cut_next_to_the_noisy_pixel():
    psi = np.array([[0.0, 2.0], [-2.0, 4.0 - 2 * np.pi]])
    assert np.abs(residues(psi)).sum() == 1  # one cut is unavoidable
    valid = np.ones((2, 2), bool)

    energy_u, cuts_u, _ = _cut_edges(psi, edge_weights("uniform", valid))
    assert energy_u == pytest.approx(1.0)
    assert len(cuts_u) == 4  # every edge is an equally cheap place for the cut

    coh = np.array([[0.9, 0.9], [0.9, 0.3]])  # pixel (1, 1) is noisy
    w = edge_weights("coherence", valid, coh, normalization="none")
    energy_c, cuts_c, edges = _cut_edges(psi, w)
    assert energy_c == pytest.approx(0.27)
    noisy = 3  # flat index of (1, 1)
    for (e,) in cuts_c:
        assert noisy in (edges.src[e], edges.dst[e])
