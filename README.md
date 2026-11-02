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
| QUBO/BQM: L2 objective, binary / one-hot encoding, masks, free gauge (default) or pinned reference (`insar_unwrap.qubo`) | implemented (API), verified against brute force on <= 3x3 fields |
| SA / SQA adapters with wall/CPU timing (`insar_unwrap.solvers`) | implemented (API), smoke-tested on 4x4 |
| Approved-dataset registry and download with size/checksum verification and provenance sidecar (`insar-unwrap datasets`, `fetch-sample`) | implemented |
| GUNW reader: native-grid crops of wrapped phase, coherence and mask, grid check by coordinates, mask decoding (`insar_unwrap.io.gunw`, `insar-unwrap inspect`) | implemented, checked on four granules (product spec 1.2.1 and 1.5.0) |
| Complex multilooking of crops (`input.multilook`, `insar_unwrap.io.multilook`) | implemented (API + config), unit-tested |
| Edge weights: `gamma_i * gamma_j` divided by the median edge weight, uniform as control arm (`insar_unwrap.qubo.weights`, ADR-001 in the wiki) | implemented (API + config), exact-solver test on a 2x2 residue |
| `insar-unwrap crop-summary`: configured crop -> validity, residues, problem descriptors (mean fringe rate, tree-integrated label span), edge weights -> BQM summary (no solving) | implemented (wiring check) |
| Metrics: label agreement modulo a constant per component, label-bound saturation, energy statistics, time-to-solution (`insar_unwrap.metrics`) | implemented (API), unit-tested |
| Experiment runner / CLI, result records, peak-memory measurement | not implemented |
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
from insar_unwrap.metrics import agreement_modulo_constant, energy_stats
from insar_unwrap.qubo import build_unwrap_bqm
from insar_unwrap.solvers import run_sa, run_sqa

field = synthesize(ramp((32, 32), 1.5, 1.0), noise_std=0.3, seed=0)
qubo = build_unwrap_bqm(field.wrapped, label_min=-2, label_max=1)  # free gauge
sa = run_sa(qubo.bqm, SaSolver(num_reads=10, num_sweeps=1000), seed=1)
sqa = run_sqa(qubo.bqm, SqaSolver(num_reads=10), seed=1)  # untuned default schedule
k, ok = qubo.decode(sa.sampleset.first.sample)
# Labels are defined up to one constant per connected component.
agree = agreement_modulo_constant(k[0], field.k_true, qubo.component, ok[0])
stats = energy_stats(sa.sampleset)
```

## Requirements

* [uv](https://docs.astral.sh/uv/) (tested with 0.9.26). uv installs the pinned
  CPython 3.12 itself; the system Python is not touched.
* Optional: Docker Engine with Compose v2 (tested with Docker 29.9 / Compose v5.6).
* Optional: `snaphu` on `PATH`. Not needed for the skeleton; `doctor` reports
  whether it is present.

Only `fetch-sample` uses the network. It needs `curl`; datasets marked
`login` in `configs/datasets.yaml` also need an Earthdata Login entry for
`urs.earthdata.nasa.gov` in `~/.netrc`. Everything else works offline.

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

`configs/experiment.example.yaml` is a complete, validating example. Its input
is the plumbing sample `nisar_sample_alos1_gunw_d00404` (a NISAR-format product
generated from ALOS-1 PALSAR data, not a NISAR observation, almost no fringes);
the HDF5 layer paths were checked on that file and on a NISAR PROVISIONAL GUNW.
SA parameters use the exact
keyword names of `dwave.samplers.SimulatedAnnealingSampler`. SQA parameters
use `PathIntegralAnnealingSampler` names (`num_reads`, `num_sweeps_per_beta`,
`Hp_field`, `Hd_field`) plus a `schedule` block (`beta`, `gamma_start`,
`gamma_end`, `num_steps`) from which `Hp_field`/`Hd_field` are generated; the
default schedule values are untuned. Unknown keys are rejected. Seeds must be
in `[0, 2**31 - 1]`, the range the samplers actually accept.

The configuration is consumed by `crop-summary` (input, multilook, mask and problem
sections); there is no experiment runner yet.

```bash
make datasets                                          # approved datasets
make fetch-sample DATASET=nisar_sample_alos1_gunw_d00404  # ~252 MiB, no login
make fetch-sample DATASET=nisar_gunw_provisional_p05023_greenland  # ~1.6 GiB, Earthdata login
make inspect FILE=data/raw/<name>/<file>.h5            # layers, grids, masks
make crop-summary SIZE=64                              # crop -> residues -> BQM size
```

## Positioning of external tools

* **SNAPHU** is treated as an external executable and a reference unwrapper
  with its own objective function. It is not a QUBO solver and is not a ground
  truth.
* **ISCE3** or other preprocessing is expected to happen outside this repo.
* **D-Wave hardware** is a possible later comparison for a few selected cases;
  the current scope is CPU-only, local samplers.

## Roadmap

1. Verify phase/sign/gauge conventions and the QUBO mapping on small synthetic
   data. (Done for tiny fields, including gauge-aware metrics.)
2. Inspect a real GUNW product; load wrapped phase, coherence and mask on one grid.
   (Done for the reader. Measurement scenes chosen 2026-11-03: NISAR
   PROVISIONAL GUNW over Greenland (ice flow) and Kumamoto (earthquake pair),
   both in the dataset registry; crops not chosen yet.)
3. Run SA vs SQA on the same QUBO at 32x32; compare images against SNAPHU.
4. Scale to 64, 128, 256; measure quality, time and memory.
5. Select a few cases for QPU comparison after confirming access terms.

## License

Copyright 2026 Syota Sasaki. Licensed under the Apache License, Version 2.0;
see [LICENSE](LICENSE).
