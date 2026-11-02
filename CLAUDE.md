# CLAUDE.md

Project notes for AI-assisted work on this repository.

## Commands

```bash
make sync            # uv sync --locked --group dev
make check           # ruff check, ruff format --check, pytest
make validate-config # insar-unwrap validate-config --config configs/experiment.example.yaml
make doctor          # insar-unwrap doctor
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
* src layout: `src/insar_unwrap/`. Keep modules small; add `io`, `qubo`, `solvers`,
  `metrics` only when their first real implementation lands.
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
* No claims of quantum advantage or speed-up over SNAPHU.
* Separate synthetic-data implementation checks from real-data evaluation.

## Scope boundaries (current)

* Local CPU only. No QPU, Braket, Amplify, GPU/CUDA, cloud credentials.
* No data download, no long computations, no QUBO solving yet.
* Never fabricate results, timings or figures for unimplemented features.
* Record implementation status honestly in README and docs.
