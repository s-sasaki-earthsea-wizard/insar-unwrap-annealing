# Experiment plan: InSAR phase unwrapping with SA and SQA

Status: design memo for a skeleton repository. Nothing below has been run.

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
| SQA (`dwave.samplers.PathIntegralAnnealingSampler`) | method under study | classical path-integral MC of a transverse-field Ising model |
| SNAPHU | conventional unwrapper | different objective (MCF / statistical cost); image-level comparison only |
| exact / strong classical solvers | optional later | e.g. brute force on tiny cases, min-cost-flow or ILP on the integer problem |

SQA is a classical Monte Carlo simulation. It is not a quantum circuit
state-vector simulation, so exponential state-vector memory estimates do not
apply to it. It also does not reproduce the non-equilibrium real-time dynamics
of a QPU. Neither quantum advantage nor speed-up over SNAPHU is claimed.

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

## 4. QUBO formulation (to be implemented and verified)

Unknowns: integer ambiguities k_i with phi_i = psi_i + 2*pi*k_i, where psi_i is
the wrapped phase in (-pi, pi].

Initial candidate objective (4-neighbour, weighted squared residual):

    E(k) = sum_{(i,j) in edges} w_ij * ( W(psi_j - psi_i) + 2*pi*(k_j - k_i) )^2
         + (optional) penalty terms for encoding constraints

where W(.) wraps to (-pi, pi] and w_ij is derived from coherence or set to 1.

Decisions to make explicit and test at implementation time:

* label range [k_min, k_max] and encoding (binary vs one-hot); number of bits
  per pixel follows from these, 2 bits/pixel is not fixed
* sign convention of the wrapped difference and of the 2*pi term
* gauge: a global shift k -> k + c leaves E unchanged; fix one reference per
  connected component of the valid-pixel graph (masking can split the graph)
* invalid pixels: drop their variables and incident edges, or down-weight
* saturation at k_min / k_max as a diagnostic that the range is too narrow
* BQM is built sparsely (dimod BQM with explicit edges). No dense N^2 matrix.
* formulation and solver invocation are separate: one BQM, many samplers.

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

## 7. Module layout (planned)

```
src/insar_unwrap/
  cli.py       argparse entry point (exists)
  config.py    pydantic schema + loader (exists)
  doctor.py    environment report (exists)
  io/          GUNW reader, grid consistency checks, synthetic generators
  qubo/        BQM construction, encodings, gauge handling
  solvers/     thin adapters: SA, SQA, SNAPHU subprocess
  metrics/     energy, saturation, gauge-aware differences, timing/memory
```

Create each module when its first real implementation lands. Avoid abstract
base classes and plugin registries until there are at least two concrete users.

## 8. Next implementation steps

1. `io/synthetic.py`: generate a smooth phase ramp + noise, wrap it, know k.
2. `qubo/build.py`: 4-neighbour BQM with configurable encoding and reference;
   unit tests that recover k on tiny fields by brute force.
3. `solvers/`: SA and SQA adapters that take a BQM and return samples + timing.
4. Inspect one real GUNW granule; replace `UNVERIFIED` paths; implement the reader.
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

To be read / confirmed before the corresponding step:

* NISAR L2 GUNW product specification (layer names, grids, mask semantics).
* SNAPHU documentation (Chen & Zebker; statistical-cost network-flow objective,
  input formats, tiling behaviour).
* D-Wave Ocean documentation for `dwave-samplers` SQA (schedule semantics of
  `Hp_field`/`Hd_field`, projection of path-integral states).
* Prior work on phase unwrapping via (quantum) annealing and on QUBO encodings
  of integer variables.
* uv documentation: project locking/syncing (`--locked`, `--frozen`) and Docker
  integration (`UV_PROJECT_ENVIRONMENT`, layer caching).

Pricing, free tiers and QPU access terms change; they are not recorded here as
fixed assumptions.
