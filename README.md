# insar-unwrap-annealing

Benchmark scaffold for InSAR phase unwrapping formulated as a QUBO and solved
with simulated annealing (SA) and simulated quantum annealing (SQA), with
SNAPHU as the conventional reference unwrapper. Target input is NISAR GUNW
wrapped interferograms.

**Status: early implementation.** The repository provides a reproducible
environment, a validated configuration schema, an environment diagnostic, a
synthetic-data generator, the sparse QUBO/BQM construction and SA/SQA sampler
adapters as a Python API. These are unit-tested on tiny synthetic fields only.
There is no experiment runner, no real-data reader and no benchmark yet, and
no performance numbers exist. See `docs/experiment-plan.md` for the research
plan and the scientific caveats that shape it.

## What is and is not implemented

| Area | Status |
|------|--------|
| uv-managed environment, lockfile, Docker image | implemented |
| `insar-unwrap validate-config` (schema + numeric consistency, no file access) | implemented |
| `insar-unwrap doctor` (versions, SA/SQA import check, SNAPHU presence) | implemented |
| Synthetic fields with exact integer truth (`insar_unwrap.io.synthetic`) | implemented (API) |
| QUBO/BQM: L2 objective, binary / one-hot encoding, masks, per-component gauge (`insar_unwrap.qubo`) | implemented (API), verified against brute force on <= 3x3 fields |
| SA / SQA adapters with wall/CPU timing (`insar_unwrap.solvers`) | implemented (API), smoke-tested on 4x4 |
| GUNW reader (HDF5 layers, grid consistency checks) | not implemented (needs a real granule) |
| Experiment runner / CLI, metrics, result records, peak-memory measurement | not implemented |
| SNAPHU runner | not implemented |
| D-Wave QPU, Braket, Fixstars Amplify, GPU | out of scope for now |

### Objective in one line

`E(k) = sum over right/down edges of w * (k_t - k_s - a)^2` with
`a = (W(psi_t - psi_s) - (psi_t - psi_s)) / 2pi`, i.e. the weighted squared
mismatch between the unwrapped gradient and the wrapped gradient, in cycles.

### SQA caveat

`PathIntegralAnnealingSampler` with its `linear`/`geometric` schedules runs
with zero transverse field (its `Gamma` argument is overwritten), which is
classical annealing. This repository always uses `beta_schedule_type="custom"`
with an explicit transverse-field schedule; the config schema rejects `Gamma`
and the non-custom SQA schedules. Details and evidence:
`docs/experiment-plan.md`, section 2.

### Python API example

```python
from insar_unwrap.config import SaSolver, SqaSolver
from insar_unwrap.io.synthetic import ramp, synthesize
from insar_unwrap.qubo import build_unwrap_bqm
from insar_unwrap.solvers import run_sa, run_sqa

field = synthesize(ramp((32, 32), 1.5, 1.0), noise_std=0.3, seed=0)
qubo = build_unwrap_bqm(field.wrapped, label_min=-2, label_max=1)
sa = run_sa(qubo.bqm, SaSolver(num_reads=10, num_sweeps=1000), seed=1)
sqa = run_sqa(qubo.bqm, SqaSolver(num_reads=10), seed=1)  # untuned default schedule
k, ok = qubo.decode(sa.sampleset.first.sample)
truth = qubo.gauge_fix(field.k_true)  # compare labels in the BQM's gauge
```

## Requirements

* [uv](https://docs.astral.sh/uv/) (tested with 0.9.26). uv installs the pinned
  CPython 3.12 itself; the system Python is not touched.
* Optional: Docker Engine with Compose v2 (tested with Docker 29.9 / Compose v5.6).
* Optional: `snaphu` on `PATH`. Not needed for the skeleton; `doctor` reports
  whether it is present.

No data download, account, API key, or network service is needed for anything
documented here.

## Local setup (uv)

```bash
make sync            # uv sync --locked --group dev
make validate-config # validates configs/experiment.example.yaml
make doctor          # environment report
make check           # ruff check + ruff format --check + pytest
```

Equivalent raw commands:

```bash
uv sync --locked
uv run --locked insar-unwrap --help
uv run --locked insar-unwrap validate-config --config configs/experiment.example.yaml
uv run --locked insar-unwrap doctor
uv run --locked ruff check . && uv run --locked ruff format --check . && uv run --locked pytest
```

If the checkout lives on a network share (CIFS/NFS), keep the virtual
environment on local disk. Either export `UV_PROJECT_ENVIRONMENT` or pass
`VENV` to make; both are honoured by every target:

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/insar-unwrap-annealing/venv
make sync check
```

Run Python through `uv run`, not through a pyenv shim: `.python-version`
contains `3.12` for uv, which pyenv does not resolve.

`--locked` fails instead of rewriting `uv.lock`. To change dependencies, edit
`pyproject.toml` and run `uv lock` deliberately.

## Docker

The image installs the same `uv.lock` and dependency group into `/opt/venv`
with `uv sync --locked` at build time. Inside the container, commands run
directly from `/opt/venv/bin`; there is no runtime re-sync, and a host `.venv`
can never shadow the container environment. The container
runs as a non-root user whose uid/gid default to 1000. `make docker-build`
passes your uid/gid as `HOST_UID`/`HOST_GID` so files written to the mounted
`results/` directory belong to you.

```bash
make docker-build    # docker compose build
make docker-check    # validate-config + doctor + pytest inside the container
docker compose run --rm app insar-unwrap doctor
```

## Configuration

`configs/experiment.example.yaml` is a complete, validating example. Items
marked `UNVERIFIED` (HDF5 dataset paths, the input file name) are placeholders
to be confirmed against a real GUNW product. SA parameters use the exact
keyword names of `dwave.samplers.SimulatedAnnealingSampler`. SQA parameters
use `PathIntegralAnnealingSampler` names (`num_reads`, `num_sweeps_per_beta`,
`Hp_field`, `Hd_field`) plus a `schedule` block (`beta`, `gamma_start`,
`gamma_end`, `num_steps`) from which `Hp_field`/`Hd_field` are generated; the
default schedule values are untuned. Unknown keys are rejected. Seeds must be
in `[0, 2**31 - 1]`, the range the samplers actually accept.

The configuration is validated but not yet consumed by a runner: label range,
encoding and reference are currently passed to `build_unwrap_bqm` directly.

## Positioning of external tools

* **SNAPHU** is treated as an external executable and a reference unwrapper
  with its own objective function. It is not a QUBO solver and is not a ground
  truth.
* **ISCE3** or other preprocessing is expected to happen outside this repo.
* **D-Wave hardware** is a possible later comparison for a few selected cases;
  the current scope is CPU-only, local samplers.

## Roadmap

1. Verify phase/sign/gauge conventions and the QUBO mapping on small synthetic
   data. (Done for tiny fields; gauge-aware metrics still to come.)
2. Inspect a real GUNW product; load wrapped phase, coherence and mask on one grid.
3. Run SA vs SQA on the same QUBO at 32x32; compare images against SNAPHU.
4. Scale to 64, 128, 256; measure quality, time and memory.
5. Select a few cases for QPU comparison after confirming access terms.

## License

MIT (see `pyproject.toml`).
