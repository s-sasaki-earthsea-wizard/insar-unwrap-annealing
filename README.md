# insar-unwrap-annealing

Benchmark scaffold for InSAR phase unwrapping formulated as a QUBO and solved
with simulated annealing (SA) and simulated quantum annealing (SQA), with
SNAPHU as the conventional reference unwrapper. Target input is NISAR GUNW
wrapped interferograms.

**Status: skeleton.** This repository currently provides a reproducible
environment, a validated configuration schema, and an environment diagnostic.
No unwrapping, QUBO construction, or benchmarking is implemented yet, and no
performance numbers exist. See `docs/experiment-plan.md` for the research
plan and the scientific caveats that shape it.

## What is and is not implemented

| Area | Status |
|------|--------|
| uv-managed environment, lockfile, Docker image | implemented |
| `insar-unwrap validate-config` (schema + numeric consistency, no file access) | implemented |
| `insar-unwrap doctor` (versions, SA/SQA import check, SNAPHU presence) | implemented |
| GUNW reader (HDF5 layers, grid consistency checks) | not implemented |
| QUBO/BQM formulation | not implemented |
| SA / SQA / SNAPHU runners, metrics, result records | not implemented |
| D-Wave QPU, Braket, Fixstars Amplify, GPU | out of scope for now |

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
to be confirmed against a real GUNW product. Solver parameters use the exact
keyword names of `dwave.samplers.SimulatedAnnealingSampler` (SA) and
`dwave.samplers.PathIntegralAnnealingSampler` (SQA); unknown keys are rejected.

## Positioning of external tools

* **SNAPHU** is treated as an external executable and a reference unwrapper
  with its own objective function. It is not a QUBO solver and is not a ground
  truth.
* **ISCE3** or other preprocessing is expected to happen outside this repo.
* **D-Wave hardware** is a possible later comparison for a few selected cases;
  the current scope is CPU-only, local samplers.

## Roadmap

1. Verify phase/sign/gauge conventions and the QUBO mapping on small synthetic data.
2. Inspect a real GUNW product; load wrapped phase, coherence and mask on one grid.
3. Run SA vs SQA on the same QUBO at 32x32; compare images against SNAPHU.
4. Scale to 64, 128, 256; measure quality, time and memory.
5. Select a few cases for QPU comparison after confirming access terms.

## License

MIT (see `pyproject.toml`).
