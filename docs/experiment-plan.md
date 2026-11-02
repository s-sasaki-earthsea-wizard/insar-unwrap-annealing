# Experiment plan: InSAR phase unwrapping with SA and SQA

Status: design memo. Synthetic generator, BQM construction and SA/SQA adapters
exist and are unit-tested on tiny fields (<= 4x4 solved, <= 32x32 built). No
benchmark, no real-data run and no SNAPHU run has been performed; no
performance numbers exist.

## 1. Goal

Measure, on NISAR GUNW wrapped interferograms, how a local simulated quantum
annealing (SQA) solver behaves on a QUBO formulation of 2-D phase unwrapping
as the problem grows from 32x32 to 64x64, 128x128 and 256x256 pixels. Record
solution quality, convergence, wall time and peak memory, and compare against
classical simulated annealing (SA) on the identical QUBO and against SNAPHU as
a conventional unwrapper.

Each size is a full image problem. Decomposition into independent 4x4 tiles is
not the baseline method. Whether 256x256 is solvable in practical time is an
open question, not a feature.

## 2. Comparison set

| Method | Role | Notes |
|--------|------|-------|
| SA (`dwave.samplers.SimulatedAnnealingSampler`) | optimizer baseline on the same QUBO | classical |
| SQA (`dwave.samplers.PathIntegralAnnealingSampler`) | method under study | classical path-integral MC of a transverse-field Ising model; always with a custom schedule, see below |
| SNAPHU | conventional unwrapper | different objective (MCF / statistical cost); image-level comparison only |
| exact / strong classical solvers | optional later | e.g. brute force on tiny cases, min-cost-flow or ILP on the integer problem |

SQA is a classical Monte Carlo simulation. It is not a quantum circuit
state-vector simulation, so exponential state-vector memory estimates do not
apply to it. It also does not reproduce the non-equilibrium real-time dynamics
of a QPU. Neither quantum advantage nor speed-up over SNAPHU is claimed.

**SQA schedule pitfall (verified 2026-11-02, dwave-samplers 1.8.0).** With
`beta_schedule_type="linear"` or `"geometric"`, `PathIntegralAnnealingSampler`
sets `Hd_field` (the transverse field schedule) to zeros, and its C++ core
overwrites the `Gamma` argument with `Hd_field[i]` at every step. No worldline
breaks are then proposed and each update is a classical heat-bath spin flip:
the run is classical annealing with random update order. Experimentally,
`Gamma` = 0.01, 1 and 50 give bit-identical samples for the same seed, with
zero breaks; a custom schedule with constant `Hd_field > 0` produces breaks.
This repository therefore always calls the sampler with
`beta_schedule_type="custom"` and an explicit schedule: fixed inverse
temperature `beta` (`Hp_field = beta`) and a transverse field decreasing
linearly from `gamma_start` to `gamma_end` (`Hd_field = beta * Gamma(s)`).
`Gamma` and the linear/geometric SQA schedules are rejected by the config
schema, and a test pins the upstream behaviour so an upgrade that changes it
is noticed. The schedule form and its default values (beta 10, Gamma 3 -> 0,
1000 steps) are an untuned initial choice.

Even with this fix, SA and SQA differ in more than quantum fluctuations: SA
uses Metropolis acceptance with sequential sweeps and an annealed beta; the
SQA core uses heat-bath updates in random order at fixed beta. Any SA/SQA
difference has to be read with that in mind.

## 3. Input data

* Primary candidate: NISAR L2 GUNW native complex wrapped interferogram, plus the
  coherence and valid/mask layers on the same grid.
* Never use the product's unwrapped phase, rewrapped, as a stand-in for the
  observed wrapped phase.
* Within one GUNW file the wrapped, coherence and unwrapped layers may live on
  different grids and resolutions. Before pairing arrays, confirm shape,
  coordinates, posting, polarization and frequency. Arrays from different grids
  must never be indexed together.
* Crops are taken on the native grid. Wrapped angles are never resampled with
  ordinary image resizing to reach a target size.
* HDF5 dataset paths are configuration, not constants. The example config
  carries `UNVERIFIED` placeholders until a real product has been inspected
  (e.g. with `h5dump -n` or h5py).
* Preprocessing (ISCE3 etc.) happens outside this repository.

## 4. QUBO formulation (implemented in `qubo/`, verified on tiny fields)

Unknowns: integer ambiguities k_i with phi_i = psi_i + 2*pi*k_i, where psi_i is
the wrapped phase in (-pi, pi].

Objective (4-neighbour, weighted squared residual). The unwrapped gradient
should match the wrapped gradient:

    E(k) = sum_{(i->j) in edges} w_ij * ( (psi_j + 2*pi*k_j) - (psi_i + 2*pi*k_i)
                                          - W(psi_j - psi_i) )^2 / (2*pi)^2
         = sum_{(i->j) in edges} w_ij * ( k_j - k_i - a_ij )^2

    a_ij = ( W(psi_j - psi_i) - (psi_j - psi_i) ) / (2*pi)   in {-1, 0, 1}

plus, for one-hot encoding only, a penalty for violated one-hot constraints.
W(.) wraps to (-pi, pi] and w_ij is derived from coherence or set to 1. This
is the L2 cost of Kelany et al. (2020), eq. (7), without their unary term.

Correction (2026-11-02): an earlier draft of this memo wrote the summand as
`( W(psi_j - psi_i) + 2*pi*(k_j - k_i) )^2`. Because |W(.)| <= pi, that
expression is minimized by k_j = k_i on every edge, i.e. it never unwraps. The
form above is the one implemented and tested.

Decisions to make explicit and test at implementation time (status in
brackets):

* label range [k_min, k_max] and encoding (binary vs one-hot); number of bits
  per pixel follows from these, 2 bits/pixel is not fixed
  [done: binary uses coefficients 1, 2, ..., 2^(B-2), c_last so every bit
  string is in range without a penalty (non-power-of-two D adds degeneracy);
  one-hot uses the assignment form with a penalty that is provably large
  enough, `1 + max_p sum_{e at p} w_e (D - 1 + |a_e|)^2`]
* sign convention of the wrapped difference and of the 2*pi term
  [done: `phase.py`; edges point right/down; tested against an independent
  oracle written from the radian definition]
* gauge: a global shift k -> k + c leaves E unchanged; fix one reference per
  connected component of the valid-pixel graph (masking can split the graph)
  [done: per component, the member nearest the component centroid, value 0 by
  default; or one explicit pixel if the graph is connected; reference bits are
  fixed and removed from the BQM]
* invalid pixels: drop their variables and incident edges, or down-weight
  [done: dropped; zero-weight edges are dropped as well]
* saturation at k_min / k_max as a diagnostic that the range is too narrow
  [open: belongs to `metrics/`]
* BQM is built sparsely (dimod BQM with explicit edges). No dense N^2 matrix.
  [done: built from coefficient arrays linear in the edge count; 256x256 with
  4 labels: 131,070 variables / 587,759 interactions (binary), built in 0.2 s;
  one-hot: 262,140 / 1,967,050 in 1.0 s. Construction only, no solving.]
* formulation and solver invocation are separate: one BQM, many samplers.
  [done: `solvers/` takes a BQM and never mutates it]

A single reference pixel is a weak gauge constraint. Shifting every pixel of
a component except its reference by +-1 costs only the summed weight of the
reference's edges (4 on a uniform interior grid), independent of image size,
as long as the shifted labels stay inside the label range. Consequences:

* label agreement with a reference solution must be measured modulo one
  constant per component (e.g. the mode of the label difference), not after
  aligning on the reference pixel alone (`UnwrapQubo.gauge_fix` does the
  latter and is only meant for exact-optimum checks);
* an energy gap of that size near the optimum is a structural feature of the
  formulation, not necessarily a poor optimizer;
* alternatives worth discussing: pinning a small patch instead of one pixel,
  a weak unary term as in Kelany et al. eq. (7), or no pin at all and
  evaluation modulo a constant.

The one-hot safe penalty (65 for 4 labels on a uniform-weight grid) is large
compared with typical edge costs (0 to a few). That compresses the energy
landscape seen by an annealer; it is a reason to keep binary as the default,
not a measured result.

A convex quadratic objective of this kind has strong classical solvers. Turning
it into a QUBO does not by itself produce any quantum benefit; this must be
stated plainly in any write-up.

## 5. Evaluation

* SA vs SQA: same BQM, same seeds policy, same budget axes (reads, sweeps).
  Metrics: BQM energy (best, mean, spread), wall time, peak RSS, fraction of
  pixels at label bounds, gauge-aware phase differences between solutions.
* vs SNAPHU: image-level comparison (residual maps, cycle counts, difference
  modulo per-component offsets). Not a ground-truth comparison.
* Rewrap agreement alone is not evidence of correct unwrapping.
* Synthetic-data checks (known truth, controlled noise) validate the code.
  Real-data runs evaluate behaviour. Keep the two in separate reports.

## 6. Reproducibility record (to implement later)

Every result should be stored with: granule id, product version, layer paths
and shapes, crop origin and size, grid metadata, mask rule, label range and
encoding, solver class and parameters, package versions (from `uv.lock`),
Git commit, seed. The skeleton does not implement this store; `results/` is
reserved for it and is not versioned.

## 7. Module layout

```
src/insar_unwrap/
  cli.py       argparse entry point (exists)
  config.py    pydantic schema + loader (exists)
  doctor.py    environment report (exists)
  phase.py     wrap(), integer gradient targets, residues (exists)
  io/          synthetic.py exists; GUNW reader + grid checks planned
  qubo/        encoding.py, build.py: BQM, encodings, gauge handling (exist)
  solvers/     annealing.py: SA and SQA adapters (exist); SNAPHU subprocess planned
  metrics/     energy, saturation, gauge-aware differences, timing/memory (planned)
```

The SA/SQA adapters record wall and CPU time of the `sample()` call only.
Per-run peak memory is not recorded yet: `ru_maxrss` is a process-wide
high-water mark, so a trustworthy per-run figure needs one subprocess per run
(planned with `metrics/`).

Create each module when its first real implementation lands. Avoid abstract
base classes and plugin registries until there are at least two concrete users.

## 8. Next implementation steps

1. [done 2026-11-02] `io/synthetic.py`: phase ramp / Gaussian bump + noise,
   wrapped, with exact integer truth and a residue count.
2. [done 2026-11-02] `qubo/build.py`: 4-neighbour BQM with configurable
   encoding and reference; exact-solver tests recover k on tiny fields and
   match brute force over integer labels (including 3 labels, noisy input).
3. [done 2026-11-02] `solvers/`: SA and SQA adapters that take a BQM and
   return samples + timing; SQA always on a custom transverse-field schedule.
4. Inspect one real GUNW granule; replace `UNVERIFIED` paths; implement the
   reader. Blocked on a locally available granule (needs an Earthdata login
   and a download, both outside the current scope) and on adding `h5py`.
5. 32x32 pilot: SA vs SQA energies and times; SNAPHU on the same crop.
6. Scale up, record metrics, decide whether 256x256 is reachable.

## 9. References (verified vs. not)

Verified in this environment (2026-11-02):

* dwave-samplers 1.8.0 exposes `PathIntegralAnnealingSampler` in
  `dwave.samplers.sqa.sampler`; `sample()` accepts `beta_range`, `num_reads`,
  `num_sweeps`, `beta_schedule_type`, `seed`, `Hp_field`, `Hd_field`, `Gamma`.
* `SimulatedAnnealingSampler.sample()` accepts `beta_range`, `num_reads`,
  `num_sweeps`, `beta_schedule_type`, `seed`, among others.
* uv 0.9.26 resolves the pinned set under CPython 3.12.12.
* dwave-samplers 1.8.0 SQA schedule semantics, from `sqa/sampler.py` and
  `sqa/src/localPIMC.cpp` plus experiments: non-custom schedules zero
  `Hd_field` and `Gamma` is overwritten (see section 2); `Hp_field` requires
  `beta_schedule_type="custom"`; with `custom`, `num_sweeps` must be omitted or
  equal `len(Hp_field) * num_sweeps_per_beta`; returned samples are projected
  classical states by default (`project_states=(True, True)`).
* Both samplers reject seeds outside `[0, 2**31 - 1]` (code checks
  `seed < 2**31`; the error text says `2^32 - 1`).
* `dimod.BinaryQuadraticModel.from_numpy_vectors` sums duplicate and reversed
  `(row, col)` entries (dimod 0.12.23).

Read (PDFs kept locally under `docs/`, not versioned):

* K. A. H. Kelany, N. Dimopoulos, C. P. J. Adolphs, B. Barabadi, A. Baniasadi,
  "Quantum annealing approaches to the phase-unwrapping problem in
  synthetic-aperture radar imaging", IEEE QCE 2020, arXiv:2010.00220. L1/L2
  integer formulations (eqs. 2, 7), binary label encoding (eq. 12),
  super-pixel decomposition into 20x20 / 16x16 / 10x10 sub-images. Their
  decomposition is not this project's baseline (full-image problems).
* K. Glatting, J. Meyer, S. Huber, G. Krieger, "Quantum optimization for phase
  unwrapping in SAR interferometry", IEEE JSTARS 18 (2025) 3488-3500,
  doi:10.1109/JSTARS.2024.3519701. L0-norm unwrapping with QAOA on <= 32
  simulated qubits; gate-model, not annealing. Useful as a reminder that L0/L1
  objectives behave differently from the L2 objective used here.
* S. K. Sekwao et al., "SAR and InSAR change detection with quantum generative
  models", arXiv:2609.05313 (2026). QCBM-based change detection; not about
  unwrapping. Not used for the current steps.

To be read / confirmed before the corresponding step:

* NISAR L2 GUNW product specification (layer names, grids, mask semantics).
* SNAPHU documentation (Chen & Zebker; statistical-cost network-flow objective,
  input formats, tiling behaviour).
* Kelany et al., IEEE TQE 3 (2022) 3101420 (journal follow-up); Otgonbaatar &
  Datcu, EUSAR 2021 (network-flow QUBO on an annealer).
* Literature on SQA schedules (e.g. Martonak, Santoro, Tosatti 2002) before
  tuning `beta` / `Gamma`.
* uv documentation: project locking/syncing (`--locked`, `--frozen`) and Docker
  integration (`UV_PROJECT_ENVIRONMENT`, layer caching).

Pricing, free tiers and QPU access terms change; they are not recorded here as
fixed assumptions.
