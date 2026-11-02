import dimod
import numpy as np
import pytest
from dwave.samplers import PathIntegralAnnealingSampler

from insar_unwrap.config import SaSolver, SqaSchedule, SqaSolver
from insar_unwrap.io.synthetic import ramp, synthesize
from insar_unwrap.metrics import agreement_modulo_constant
from insar_unwrap.qubo import build_unwrap_bqm
from insar_unwrap.solvers import run_sa, run_sqa, sqa_schedule

# Small budgets: these tests check plumbing, not solver quality.
SA_CFG = SaSolver(num_reads=4, num_sweeps=200)
SQA_CFG = SqaSolver(num_reads=4, schedule=SqaSchedule(beta=5.0, gamma_start=2.0, num_steps=200))


@pytest.fixture(scope="module")
def tiny():
    field = synthesize(ramp((4, 4), 1.5, -1.0))
    return field, build_unwrap_bqm(field.wrapped, label_min=-1, label_max=2)


@pytest.mark.parametrize(("runner", "cfg"), [(run_sa, SA_CFG), (run_sqa, SQA_CFG)])
def test_runner_solves_tiny_noise_free_field(tiny, runner, cfg):
    field, qubo = tiny
    before = qubo.bqm.copy()
    run = runner(qubo.bqm, cfg, seed=1)
    assert qubo.bqm == before  # one BQM, many samplers: never mutated
    assert run.best_energy == pytest.approx(0.0)
    assert np.allclose(run.energies, qubo.bqm.energies(run.sampleset))
    k, ok = qubo.decode(run.sampleset.first.sample)
    assert ok.all()
    assert agreement_modulo_constant(k[0], field.k_true, qubo.component).fraction == 1.0
    assert run.wall_time_s > 0 and run.cpu_time_s >= 0
    assert len(run.sampleset) == cfg.num_reads


@pytest.mark.parametrize(("runner", "cfg"), [(run_sa, SA_CFG), (run_sqa, SQA_CFG)])
def test_runner_is_reproducible_by_seed(tiny, runner, cfg):
    _, qubo = tiny
    a = runner(qubo.bqm, cfg, seed=5).sampleset.record.sample
    b = runner(qubo.bqm, cfg, seed=5).sampleset.record.sample
    assert np.array_equal(a, b)


def test_sqa_always_uses_custom_schedule_with_transverse_field(tiny):
    _, qubo = tiny
    run = run_sqa(qubo.bqm, SQA_CFG, seed=0)
    assert run.params["beta_schedule_type"] == "custom"
    assert "Gamma" not in run.params
    hd = np.asarray(run.params["Hd_field"])
    assert hd[0] == pytest.approx(5.0 * 2.0) and hd[-1] == pytest.approx(0.0)
    assert len(run.params["Hp_field"]) == SQA_CFG.total_sweeps


def test_sqa_explicit_fields_take_precedence(tiny):
    _, qubo = tiny
    cfg = SqaSolver(num_reads=2, Hp_field=[3.0] * 50, Hd_field=list(np.linspace(6, 0, 50)))
    run = run_sqa(qubo.bqm, cfg, seed=0)
    assert run.params["Hp_field"] == [3.0] * 50


def test_sqa_schedule_shape():
    hp, hd = sqa_schedule(SqaSchedule(beta=2.0, gamma_start=4.0, gamma_end=1.0, num_steps=4))
    assert hp.tolist() == [2.0] * 4
    assert hd.tolist() == pytest.approx([8.0, 6.0, 4.0, 2.0])


@pytest.mark.parametrize(("runner", "cfg"), [(run_sa, SA_CFG), (run_sqa, SQA_CFG)])
def test_seed_outside_sampler_range_is_rejected(tiny, runner, cfg):
    with pytest.raises(ValueError, match="seed"):
        runner(tiny[1].bqm, cfg, seed=2**31)


def test_upstream_default_sqa_schedule_has_no_transverse_field():
    """Pin the dwave-samplers behaviour that motivates the custom schedule.

    With beta_schedule_type="geometric" the sampler zeroes Hd_field and ignores
    Gamma, so no worldline breaks occur and results do not depend on Gamma. If
    an upgrade changes this, revisit insar_unwrap.solvers.annealing.
    """
    rng = np.random.default_rng(0)
    h = {i: float(rng.normal()) for i in range(20)}
    j = {(i, i + 1): float(rng.normal()) for i in range(19)}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, j)
    sampler = PathIntegralAnnealingSampler()
    kw = {"num_reads": 3, "num_sweeps": 50, "seed": 3, "project_states": (True, False)}
    lo = sampler.sample(bqm, Gamma=0.01, **kw)
    hi = sampler.sample(bqm, Gamma=50.0, **kw)
    assert int(np.sum(lo.info["num_breaks"])) == 0
    assert int(np.sum(hi.info["num_breaks"])) == 0
    assert np.array_equal(lo.record.sample, hi.record.sample)
    custom = sampler.sample(
        bqm,
        num_reads=3,
        seed=3,
        beta_schedule_type="custom",
        Hp_field=np.full(50, 2.0),
        Hd_field=np.full(50, 4.0),
        project_states=(True, False),
    )
    assert int(np.sum(custom.info["num_breaks"])) > 0
