# Native MPS Liouvillian spectrum solver

For a dedicated trace-constrained NESS solve, use `NessConfig` and
`solve_ness`. Its GPU-resident two-site least-squares sweeps, fixed-number
Bose--Hubbard example, and CPU/GPU validation are documented in
[the NESS guide](../../../projects/ness_mps/README.md).

`solve_liouvillian` accepts a native Renormalizer Liouvillian MPO and MPS guesses,
and returns capped MPS eigenoperators with original-operator residuals. It can
also solve for a trace-normalized NESS. CPU and GPU use the same API.

This is a reusable research implementation with independent validation, not a
claim of universal convergence, certified global spectral coverage, or a
demonstrated GPU speedup. Read the
[validation record](../../../projects/liouvillian_spectrum/VALIDATION.md).

## Usage

```python
from renormalizer.mps.liouvillian import (
    SpectrumConfig, ShiftedSolveConfig, solve_liouvillian,
)

# L: native Mpo, psi: native Mps, both with the same local operator basis.
settings = SpectrumConfig(
    bond_cap=64,
    covariance_rank=4,
    max_subspace=40,
    max_outer=12,
    shifts_per_outer=4,
    tolerance=1e-7,
    inner=ShiftedSolveConfig(
        max_sweeps=8,
        local_restart=40,
        local_max_cycles=4,
        tolerance=1e-8,
    ),
)
result = solve_liouvillian(
    L, [psi], n_modes=3, trace="paired", compute_ness=True, config=settings,
)

# NESS first (eigenvalue zero), followed by the requested decay modes.
eigenvalues = result.all_eigenvalues
eigenoperators = result.all_eigenstates  # list of ordinary native Mps
residuals = result.all_residuals

# Inspect before treating the requested accuracy as achieved.
print(result.status, result.converged, residuals)
# Metadata contains no dense states and can be written directly with json.dump.
metadata = result.diagnostics()
```

`trace="paired"` explicitly asserts a site-local `|ket,bra>` vectorization with
local operator dimension `d*d`. It builds the vectorized identity from local
diagonal entries. For other layouts or charge-restricted identities, supply an
MPS `trace` satisfying `inner(trace, x) == Tr(x)`. An arbitrary MPO does not
contain enough metadata to infer this functional. The solver checks `L† trace`.

With a trace functional, `n_modes` counts nonstationary modes and the seeds are
projected into the trace-free space. This mode assumes a unique stationary
state and no additional nondecaying modes in that sector. With `trace=None`,
`n_modes` counts the rightmost modes of the supplied operator/sector, including
any zero modes. NESS normalization requires a trace functional.

`seeds` may be one MPS or a sequence. All must have the same conserved total
quantum number; repeat separate calculations for different sectors. Without
seeds, native random total-charge-zero MPS are constructed. Supply independent
guesses for degenerate clusters and nontrivial charge sectors. An exact missing
seed component cannot be manufactured by a rational spectral transformation.

## Core algorithm

1. Normalize and truncate seed MPS, and remove trace when requested.
2. Store a covariance as `X = B B†`, using only a short list of capped MPS
   factors. For numerical stability use the auxiliary operator `A = L - s I`,
   where `s = stability_shift > 0`, on the current projected space.
3. Approximately solve `A Xnew + Xnew A† = -B B†` by low-rank ADI. For a
   right-half-plane shift `z` and `a = Re(z) > 0`, update

   ```text
   (z I - A) Y = W
   Z <- [Z, sqrt(2*a) Y]
   W <- truncate(W - 2*a*Y)
   ```

   Each column of `Y` is obtained by nonsymmetric two-site DMRG sweeps with
   matrix-free local GMRES and a bond-truncating SVD after every center update.
   Warm starts reuse the preceding shifted solution.
4. Check the true represented-factor Lyapunov residual. Compress covariance
   factors through their physical Gram matrix, then truncate their actual MPS
   representations and normalize the source for the next outer cycle.
5. Incrementally contract `G = V† V` and `H = V† L V` for the dictionary `V`.
   Whiten `G` and solve only the small projected spectral problem using Schur
   decomposition. Materialize and truncate the requested rightmost Ritz MPS.
6. Check every returned candidate against the original MPO. Soft-lock accurate
   modes for subsequent discovery, retain neighboring guard modes, and use
   original-operator inverse refinement. Dictionary restarts rebuild physical
   capped MPS and check preservation of the retained subspace and residuals.

For an eigenvalue `lambda = -gamma + i*omega`, the exact auxiliary inverse
amplifies the diagonal covariance direction by `1 / (2*(s + gamma))`.
Oscillation frequency cancels. This is the real-part targeting mechanism;
sorting projected eigenvalues alone does not discover missing modes. The
underlying Lyapunov inverse-iteration construction is established work:
[Elman and Wu](https://www.cs.umd.edu/~elman/papers/lyap-rightmost.pdf).

Fixed-budget inner cycles are deliberately inexact. A cycle with
`inverse_step_resolved=False` did not meet `lyapunov_tolerance`; its source
update is heuristic guidance, not an accurate application of the exact lifted
inverse. Large covariance tails, inaccurate inverse solves, or truncation can
defeat the theoretical targeting. All are recorded in the history.

Soft-locking uses the orthogonal projector onto the complement of the current
accurate modes. For an invariant locked block, the projected original
resolvent is the resolvent of the complementary compression. Approximate
locking and truncation are measured by a fresh projected shifted residual.
The full original MPO remains the source of final eigenvalues and residuals.

The NESS uses the augmented nonsymmetric equation
`(-L + alpha |b><trace|) rho = alpha*b`, with unit-trace reference `b` and
`alpha = ness_scale`. The rank-one term is contracted locally, not constructed
as a global dense matrix. The capped result is trace-normalized and its
original zero-eigenvalue residual is measured. Neither uniqueness nor
positivity is inferred from a small numerical residual.

## What was learned from DMRG[FEAST]

[Baiardi, Kelemen and Reiher, Excited-state DMRG made simple with FEAST](https://arxiv.org/pdf/2110.00092)
build their subspace using MPS solutions of shifted MPO equations, obtained
through local GMRES sweeps, followed by a small Rayleigh–Ritz problem. Their
Eqs. (14), (19), and (22) motivate this inner kernel and its bounded local
iterations. Here Lyapunov ADI replaces the contour quadrature. The Hermitian
half-contour simplification and variational arguments are not transferred to
a generic nonnormal Liouvillian. Global residual checks follow truncation.

## Representation and memory contract

- The production package contains no `todense`, `from_dense`, global Kronecker
  construction, or full-space dense/sparse eigensolver fallback.
- Every stored inverse iterate, ADI recurrence factor, refreshed covariance
  factor, restarted state, and returned eigenoperator obeys `bond_cap`.
- The original MPO and exact trace functional are fixed operator metadata.
  They are not truncated by the eigensolver. The NESS reference RHS is capped.
- The dictionary does not cache enlarged MPO-applied MPS. It contracts original
  MPO matrix elements directly and updates only new rows/columns.
- True residual audits temporarily form exact MPO images and direct-sum MPS
  with larger bonds, then QR-canonicalize them to avoid subtractive cancellation.
  Those temporary objects are not dense full-space arrays and are not retained
  as iteration states. Reserve memory for them: `bond_cap` is a retained-state
  limit, not a bound on every transient tensor workspace.
- Dense spectral solves are limited to the configured subspace and covariance
  Gram matrices. Local tensor QR/SVD and small GMRES Hessenberg problems are
  also necessary, as in standard DMRG.

The spectrum of a single covariance Gram matrix is not a list of decay rates.
Covariance rank, per-MPS bond dimension, and dictionary width are separate
limits. Returned-state physical Gram checks prevent falsely counting duplicate
MPS created by nonlinear truncation as independent eigenmodes.

## Main controls and outputs

| Setting | Meaning |
|---|---|
| `bond_cap`, `cutoff` | Retained MPS bond limit and native relative singular-value threshold |
| `covariance_rank` | Number of retained covariance factors |
| `max_subspace` | Maximum number of stored dictionary MPS |
| `max_outer`, `shifts_per_outer` | Outer-cycle and per-cycle ADI-shift budgets |
| `poles` | Optional explicit right-half-plane ADI shifts; no exact spectrum required |
| `damping`, `frequency_window` | Positive initial shift scale and optional exploratory frequency interval |
| `stability_shift` | Positive auxiliary shift; final eigenvalues use the original MPO |
| `tolerance` | Absolute original eigen-residual, normalized by state norm, in generator units |
| `lyapunov_tolerance` | Relative true residual defining a resolved auxiliary inverse step |
| `gram_cutoff` | Relative numerical-rank threshold for small overlap matrices |
| `guard` | Additional neighboring modes retained during extraction/restarts |
| `refinement_steps` | Budget for original-operator inverse refinement |
| `restart_tolerance` | Allowed retained-subspace loss and residual increase during restart |
| `min_outer` | Minimum discovery cycles before the early outer convergence exit |
| `max_seconds` | Cooperative time budget checked between outer/refinement stages; an active solve may overrun it |
| `ness_scale` | Coefficient of the NESS trace augmentation |
| `inner` | `ShiftedSolveConfig`: sweep, local Krylov, tolerance, stagnation and backend controls |

The outer `bond_cap` and `cutoff` override the corresponding inner settings so
all retained states share one clear cap. `max_sweeps` counts complete
forward/backward pairs. Local Krylov storage is bounded by `local_restart`.

`SpectrumResult` exposes `eigenvalues`, `eigenstates`, `residuals`, `ness`,
`ness_residual`, `ness_trace_error`, `status`, `converged`, `history`, and
`diagnostics()`. The `all_*` properties prepend the NESS when requested.
History includes actual inner residuals, truncation diagnostics, work counts,
covariance tails, physical dictionary width and observed bonds.

`converged` means that the requested number of physically independent returned
MPS passes the specified residual and trace checks. It does not prove that no
unseeded slower mode exists; `coverage_certified` remains false. Treat statuses
such as `dictionary_budget`, `outer_budget`, `ness_not_converged`, or
`output_rank_deficient` as incomplete results.

## CPU, GPU, and reproduction

Select the process backend **before importing Renormalizer**. CPU jobs use
`CUDA_VISIBLE_DEVICES=""`. GPU jobs use `RENO_REQUIRE_GPU=1` and `RENO_GPU=0`;
failure to initialize CUDA then raises instead of silently using CPU.
`ShiftedSolveConfig(backend="cpu" or "gpu")` checks the actual backend.

The local GMRES vectors, local MPO contractions, and symmetry-block center SVD
stay on the selected NumPy/CuPy backend. Small Hessenberg and spectral matrices
use host LAPACK. Native Renormalizer still stores persistent MPS cores on the
host; this is GPU-accelerated local work, not fully device-resident MPS storage.

From the Renormalizer repository root on this cluster:

```bash
sbatch projects/liouvillian_spectrum/slurm/test_cpu.sbatch
sbatch projects/liouvillian_spectrum/slurm/test_gpu.sbatch
sbatch projects/liouvillian_spectrum/slurm/example_cpu.sbatch
sbatch projects/liouvillian_spectrum/slurm/example_gpu.sbatch
```

The [example driver](../../../projects/liouvillian_spectrum/examples/run_spin_chain.py)
shows the complete native-MPO workflow and writes JSON diagnostics. The
[cap comparison](../../../projects/liouvillian_spectrum/examples/benchmark_caps.py)
uses identical physical inputs for CPU/GPU or different bond caps.
Cluster scripts include installation-specific Python/vendor paths.

As of 2026-09-14, **21 distinct CPU tests have passed**. The two-spin cap-four
example obtained maximum eigen-residual `4.47e-12` and NESS residual `4.76e-14`.
The deliberately insufficient caps and work budgets correctly report failure.
GPU test job **290684** is queued for resources, so actual GPU execution and
CPU/GPU agreement remain unverified. See the validation record for job IDs,
independent oracles, sector scope, logs, and exact limitations.
