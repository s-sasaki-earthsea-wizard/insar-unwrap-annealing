# CLAUDE.md

Project notes for AI-assisted work on this repository.

## Commands

```bash
make sync            # uv sync --locked --group dev
make check           # ruff check, ruff format --check, pytest
make validate-config # insar-unwrap validate-config --config configs/experiment.example.yaml
make doctor          # insar-unwrap doctor
make datasets / make fetch-sample DATASET=<name> / make inspect FILE=<h5>
make crop-summary [SIZE=n]  # configured crop -> residues -> BQM summary, no solving
make docker-build && make docker-check
```

Locally, always use `uv run --locked`; never let a routine command rewrite
`uv.lock`. In Docker the lock is enforced at build time and commands run
straight from `/opt/venv/bin`.
Do not modify the system Python. Do not create `.venv` inside the Docker image
(it lives at `/opt/venv`).

## Conventions

* Identifiers, comments, docstrings, commit messages: English. Google-style docstrings.
* PEP 8 via ruff (line length 100). `snake_case` functions, `PascalCase` classes.
* src layout: `src/insar_unwrap/`. Keep modules small. Existing: `phase.py`,
  `io/synthetic.py`, `qubo/{encoding,build}.py`, `solvers/annealing.py`,
  `metrics/{labels,energy}.py`, `io/{datasets,gunw}.py`. Add per-run memory
  measurement in `metrics` only when its first real implementation lands.
* Never call the pyenv `python3` shim (pyenv has no 3.12 for `.python-version`
  and the shim hangs); use `uv run --locked python`. The checkout is on local
  disk with an in-repo `.venv`; `VENV=` / `UV_PROJECT_ENVIRONMENT` are only for
  checkouts on a network share.
* Tests cover entry points and boundaries, not exhaustive matrices.
* Session notes go to `.claude-notes/` (not versioned).

## Scientific guardrails

* SA and SQA must receive the same BQM built from the same input; compare them
  on energy, variance, time and memory, not on "correctness" alone.
* SNAPHU has a different objective. Compare image-level results; never call it
  ground truth. Product unwrapped phase is not ground truth either.
* Rewrap agreement is a necessary, not sufficient, check.
* Full-image problems per size (32, 64, 128, 256). Tiling into 4x4 blocks is not
  the default method. 256x256 feasibility is unverified.
* Integer label range, encoding and reference/gauge handling are configurable;
  2 bits/pixel is an initial choice, not an assumption.
* Keep the BQM sparse. Never allocate a dense N^2 matrix by default.
* SQA here means `dwave.samplers.PathIntegralAnnealingSampler`, a classical
  path-integral Monte Carlo. Do not describe it as a state-vector simulation,
  do not apply state-vector memory limits to it, and do not treat it as a
  faithful model of QPU dynamics. `SimulatedAnnealingSampler` is SA, not SQA.
* SQA must run with `beta_schedule_type="custom"` and `Hd_field > 0` somewhere.
  The sampler's linear/geometric schedules zero the transverse field and ignore
  `Gamma` (dwave-samplers 1.8.0), i.e. they are classical annealing. Do not
  reintroduce `Gamma` or non-custom SQA schedules in the config.
* Objective is `sum w * (k_t - k_s - a)^2` with
  `a = (W(dpsi) - dpsi) / 2pi`; not `(W(dpsi) + 2pi dk)^2` (that never unwraps).
* Compare label fields modulo one constant per component
  (`metrics.agreement_modulo_constant`). The default gauge is free
  (`reference_mode="none"`); pinned modes are for exact tests and as a
  comparison arm. A single reference pixel is a weak gauge pin (bulk shift
  costs only the reference's edge weights).
* No claims of quantum advantage or speed-up over SNAPHU.
* Separate synthetic-data implementation checks from real-data evaluation.

## Scope boundaries (current)

* Local CPU only. No QPU, Braket, Amplify, GPU/CUDA, cloud credentials.
* Data download: only datasets listed in `configs/datasets.yaml`, via
  `fetch-sample`, into `data/raw/` (never versioned, never deleted by automated
  work). Adding an entry needs Syota's approval. Approved: D00404 (NISAR-format
  sample generated from ALOS-1 PALSAR data; not a NISAR observation, never
  describe it as one; plumbing only) and P05023 (NISAR PROVISIONAL GUNW, Los
  Angeles, Earthdata login via `~/.netrc`; approved 2026-11-03). Never print
  or copy credentials.
* No long computations. Tests solve only tiny problems (<= 4x4); smoke runs
  beyond that are not benchmarks and are not recorded as results.
* Never fabricate results, timings or figures for unimplemented features.
* Record implementation status honestly in README and docs.
