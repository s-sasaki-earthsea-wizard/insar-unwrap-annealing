# Experiment plan: InSAR phase unwrapping with SA and SQA

Status: design memo. Synthetic generator, BQM construction, SA/SQA adapters
and solution metrics exist and are unit-tested on tiny fields (<= 4x4 solved, <= 32x32 built). No
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
* HDF5 dataset paths are configuration, not constants. Verified on two
  granules (2026-11-02/03): D00404 (product spec 1.2.1, missionId ALOS) and
  P05023 (spec 1.5.0, missionId NISAR). Both use
  `/science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/{HH/wrappedInterferogram,
  HH/coherenceMagnitude, mask}` on a 20 m grid (EPSG 32611 for both), and an
  80 m `unwrappedInterferogram` grid whose top-left corner coincides with the
  20 m grid's (centres offset by 30 m), so it nests 4:1.
* Mask encodings differ: spec 1.2.1 stores a uint8 three-digit code
  `<water><reference subswath><secondary subswath>`; spec 1.5.0 stores uint32
  with the same code in bits 0-7 and RSLC anomaly flags in bits 8-23. Fill is
  255 in both. `io.gunw.usable_mask` handles both (fill, zero subswath digit,
  anomaly bits; water optional). P05023 has no anomaly bits set.
* `io.gunw.read_crop` checks that the coherence and mask layers sit under a
  group whose coordinate arrays equal those of the wrapped layer, then reads
  only the requested window.
* Fringe density at the native 20 m posting is low in both granules (block
  statistics over 256x256 windows, deterministic gradient from summed
  neighbour products; quick look, not a result): D00404 median 0.04, max 0.37
  fringes per 32 px; P05023 median 0.015, max 0.06 per 32 px and at most 0.5
  per 256 px. After 4x4 complex multilooking P05023 reaches at most 0.25 per
  32 px and 2 per 256 px. Small native crops of such scenes contain at most a
  wrap line or noise-induced wraps; dense fringes need strong deformation
  gradients. Whether to multilook before cropping is an open decision.
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
* gauge: a global shift k -> k + c leaves E unchanged on each connected
  component of the valid-pixel graph (masking can split the graph)
  [done: default `none` leaves the gauge free and labels are compared modulo
  one constant per component (`metrics.agreement_modulo_constant`), decided
  2026-11-02. Pinned modes kept for exact-optimum tests and as a comparison
  arm: `per_component` pins the member nearest each component centroid,
  `pixel` one explicit pixel if the graph is connected; value 0 by default;
  reference bits are fixed and removed from the BQM]
* invalid pixels: drop their variables and incident edges, or down-weight
  [done: dropped; zero-weight edges are dropped as well]
* saturation at k_min / k_max as a diagnostic that the range is too narrow
  [done: `metrics.bound_saturation`; with a free gauge, labels on both
  bounds are the stronger warning sign]
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
  aligning on the reference pixel alone (`metrics.agreement_modulo_constant`
  does the former; `UnwrapQubo.gauge_fix` does the latter, only for pinned
  BQMs, and is meant for exact-optimum checks);
* an energy gap of that size near the optimum is a structural feature of the
  formulation, not necessarily a poor optimizer;
* annealers can get stuck in that bulk-shifted state (seen in one untuned SQA
  smoke run, not a measured rate).

Gauge options considered (2026-11-02):

| Option | Effect | Status |
|--------|--------|--------|
| A. single reference pixel per component | optimum unchanged; bulk shift costs only the reference's edges; centroid choice ignores coherence | implemented; kept for exact tests and as a comparison arm |
| B. pinned patch | shift cost = patch perimeter; needs pre-unwrapped patch labels, wrong if residues inside | not planned |
| C. unary term `w * sum k^2` (Kelany eq. 7) | no cheap bulk shift; changes the optimum unless `w` is below the unknown gradient-cost gap (about 1/N), and is then too weak at finite temperature | only if D proves problematic |
| D. no pin, evaluate modulo a constant per component | objective unchanged, full label range usable; degenerate ground state, so equality checks must be modulo a constant | default (`reference_mode="none"`) |

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
* Comparison protocol (decided 2026-11-02):
  * primary: energy versus CPU time per problem size, over a log-spaced grid
    of anneal lengths (sweeps / schedule steps);
  * time-to-solution only where the target energy is known: residue-free
    synthetic fields, where the free-gauge optimum is E* = 0;
  * equal-sweep tables as a secondary view only. One SQA sweep updates every
    Trotter slice and costs far more than one SA sweep, and the update rules
    differ (Metropolis sequential vs heat-bath random order);
  * control arm: the same `PathIntegralAnnealingSampler` code with
    `Hd_field = 0` and an annealed `Hp_field`. This is classical heat-bath
    annealing and isolates the transverse-field effect from implementation
    differences. It is never labelled SQA.
* Tuning protocol (decided 2026-11-02): SQA starts from a fixed beta and a
  linearly decreasing Gamma; (beta, Gamma_start, steps) and SA's
  (beta_range, sweeps) are tuned on synthetic tuning seeds that are disjoint
  from evaluation seeds, with the same budget for both solvers. The first
  target is 32x32; 256x256 handling is deliberately left open. Caveat: Gamma
  acts uniformly on all bits while binary label coefficients differ, so
  high-order bits see larger local fields.
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
  metrics/     labels.py (agreement modulo a constant, saturation), energy.py
               (energy statistics, TTS) exist; per-run memory planned
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
   reader. First input (approved 2026-11-02): the pre-launch NISAR-format
   sample GUNW generated from ALOS-1 PALSAR data (composite release D00404, no
   login needed). It is not a NISAR observation and must be described as
   such. A quick look showed almost no fringes (the product's unwrapped phase
   spans less than one cycle over the whole frame), so this granule is used
   only to build and check the reader, masks and weights (decided
   2026-11-02). Real-data measurements will use a fringe-bearing product
   chosen after a new search; products that need an Earthdata login need a
   separate decision. [done 2026-11-03: registry + `fetch-sample`, `inspect`,
   `io.gunw.read_crop`, `crop-summary`; P05023 (login) approved and checked;
   coherence-based edge weights still open]
5. [done 2026-11-02, except per-run peak memory] `reference.mode: none` and
   `metrics/` (comparison modulo a constant per component, energy statistics,
   time-to-solution, label-bound saturation).
6. 32x32 pilot: SA vs SQA (and the Hd = 0 control arm) energies and times;
   SNAPHU on the same crop once its installation is decided.
7. Scale up, record metrics; how to approach 256x256 is decided later.

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
