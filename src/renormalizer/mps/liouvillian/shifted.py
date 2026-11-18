"""Truncated, matrix-free MPS solutions of nonsymmetric shifted systems.

``solve_shifted`` solves ``(z I - L + alpha |u><v|) x = b`` by two-site
alternating Galerkin sweeps.  The optional separable term permits a trace
constraint for a stationary state without assembling another MPO.  Every
local operator application is a tensor contraction, and every local solve is
restarted GMRES.  No effective operator matrix is materialized.

Local Krylov vectors, contractions and symmetry-block SVDs use Renormalizer's
NumPy/CuPy backend.  Only the small GMRES Hessenberg system and quantum-number
metadata use NumPy.  Renormalizer stores persistent site tensors on the host;
this module keeps the active local solve on the device, not the entire chain.

A direct Galerkin sweep is not a globally convergent least-squares method for
an arbitrary nonnormal operator.  The reported convergence always refers to
the original, post-truncation global residual, never the projected residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np

from renormalizer import Mpo, Mps
from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.lib import Environ
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.utils import CompressConfig, CompressCriteria

from .tensor import absorb_coefficient, combine, norm as stable_norm


@dataclass(frozen=True)
class ShiftedSolveConfig:
    """Work and truncation controls for one shifted linear system.

    ``bond_cap`` caps each retained MPS bond. ``cutoff`` is a relative
    individual-singular-value threshold, matching Renormalizer's convention;
    it is not a bound on the total discarded norm. ``max_sweeps`` counts
    complete forward/backward pairs. ``local_restart`` bounds local Krylov
    storage, and ``local_max_cycles`` bounds restarted GMRES cycles per center.
    ``local_tolerance`` and ``tolerance`` are relative local/global residual
    tolerances. ``stagnation_sweeps=0`` disables early stagnation stopping.
    ``backend`` validates the backend selected when Renormalizer was imported;
    it cannot change that process-wide selection.
    """

    bond_cap: int = 64
    cutoff: float = 1e-12
    max_sweeps: int = 8
    local_restart: int = 40
    local_max_cycles: int = 4
    local_tolerance: float = 1e-10
    tolerance: float = 1e-8
    stagnation_sweeps: int = 3
    stagnation_tolerance: float = 1e-3
    backend: str = "auto"

    def __post_init__(self):
        for name in ("bond_cap", "max_sweeps", "local_restart", "local_max_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.cutoff < 1:
            raise ValueError("cutoff must lie strictly between zero and one")
        for name in ("local_tolerance", "tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.stagnation_sweeps < 0 or int(self.stagnation_sweeps) != self.stagnation_sweeps:
            raise ValueError("stagnation_sweeps must be a nonnegative integer")
        if not 0 <= self.stagnation_tolerance < 1:
            raise ValueError("stagnation_tolerance must lie in [0, 1)")
        if self.backend not in ("auto", "cpu", "gpu"):
            raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")


@dataclass(frozen=True)
class RankOneTerm:
    """Separable addition ``alpha |u><v|``; ``u`` and ``v`` are MPS vectors."""

    alpha: complex
    u: Mps
    v: Mps


@dataclass
class ShiftedSolveReport:
    """Measured diagnostics; ``residual`` certifies the returned best iterate.

    ``discarded_norms`` records the relative local norm discarded by each
    two-site SVD. It measures truncation of that center, not accumulated global
    error. ``matvecs`` counts local effective-operator applications, including
    initial and final residual checks in GMRES.
    """

    converged: bool
    residual: float
    sweeps: int
    matvecs: int
    max_bond: int
    backend: str
    shift: complex
    stop_reason: str
    seconds: float
    local_solves: int = 0
    local_iterations: int = 0
    local_failures: int = 0
    max_local_dimension: int = 0
    max_local_residual: float = 0.0
    residual_history: list[float] = field(default_factory=list)
    discarded_norms: list[float] = field(default_factory=list)


def _inner(left: Mps, right: Mps) -> complex:
    return complex(left.coeff).conjugate() * complex(right.coeff) * left.conj().dot(right)


def _norm(state: Mps) -> float:
    value = complex(_inner(state, state))
    if not np.isfinite(value) or value.real < -1e-12 * max(abs(value), 1.0):
        raise FloatingPointError(f"invalid MPS squared norm: {value}")
    return float(np.sqrt(max(value.real, 0.0)))


def _absorbed(state: Mps) -> Mps:
    return absorb_coefficient(state).to_complex()


def _same_space(left, right) -> bool:
    return left.site_num == right.site_num and tuple(left.pbond_list) == tuple(right.pbond_list)


def shifted_residual(mpo: Mpo, rhs: Mps, solution: Mps, shift: complex,
                     rank_one: RankOneTerm | None = None) -> float:
    """Evaluate the untruncated global relative residual using MPS algebra.

    A temporary exact MPO application and direct-sum residual are QR
    canonicalized before their norm is contracted. This avoids cancellation
    in a difference of nearly equal scalar overlaps and never forms ``L†L``.
    Temporary bond dimensions can exceed ``bond_cap``; these states are used
    only for certification and are not retained by the iteration.
    """
    denominator = stable_norm(rhs)
    if denominator == 0:
        raise ValueError("the shifted-system right-hand side must be nonzero")
    work = _absorbed(solution)
    states = [rhs, mpo.apply(work), work]
    weights = [1.0, 1.0, -complex(shift)]
    if rank_one is not None and rank_one.alpha != 0:
        states.append(rank_one.u)
        weights.append(-complex(rank_one.alpha) * _inner(rank_one.v, work))
    # The shared algebra explicitly handles one-site chains, zero MPO images,
    # and exact cancellation, which native Mps.add/canonicalise do not handle.
    residual = combine(states, weights, bond_cap=None)
    return stable_norm(residual) / denominator


class _ConjugatedSites:
    def __init__(self, state):
        self.state = state

    def __getitem__(self, site):
        return self.state[site].conj()


def _project(source, solution, identity, environment, center, methods):
    left_method, right_method = methods
    left = environment.GetLR("L", center[0] - 1, source, identity,
                             method=left_method, mps_conj=_ConjugatedSites(solution))
    right = environment.GetLR("R", center[-1] + 1, source, identity,
                              method=right_method, mps_conj=_ConjugatedSites(solution))
    source_center = asxp(source[center[0]])
    if len(center) == 2:
        source_center = xp.tensordot(source_center, asxp(source[center[1]]), axes=(-1, 0))
    expression = hop_expr(left, right, [asxp(identity[i]) for i in center],
                          source_center.shape)
    return expression(source_center)


def _gmres(apply: Callable, rhs, initial, restart: int, max_cycles: int, tolerance: float):
    """Restarted complex GMRES with device-resident large arrays.

    Two passes of classical Gram--Schmidt use BLAS matrix-vector operations.
    Givens rotations and the at-most-``restart`` triangular solve stay on the
    host. A recurrence residual only triggers a candidate check; convergence
    is decided by an explicit matvec residual.
    """
    vector = initial.copy()
    rhs_norm = float(xp.linalg.norm(rhs).item())
    target = tolerance * rhs_norm
    # The homogeneous local problem has an exact zero solution. The caller
    # handles a zero center without allowing a zero full MPS into a sweep.
    if rhs_norm == 0:
        return xp.zeros_like(rhs), 0.0, 0, 0, True
    dimension = rhs.size
    restart = min(restart, dimension)
    basis = xp.empty((dimension, restart + 1), dtype=rhs.dtype)
    iterations = matvecs = 0
    tiny = np.finfo(float).tiny
    residual_norm = float("inf")
    for _ in range(max_cycles):
        residual = rhs - apply(vector)
        matvecs += 1
        beta = float(xp.linalg.norm(residual).item())
        if beta <= target:
            return vector, beta / rhs_norm, iterations, matvecs, True
        if not np.isfinite(beta):
            raise FloatingPointError("nonfinite local GMRES residual")
        basis[:, 0] = residual / beta
        hessenberg = np.zeros((restart + 1, restart), dtype=np.complex128)
        cosines = np.zeros(restart)
        sines = np.zeros(restart, dtype=np.complex128)
        projected_rhs = np.zeros(restart + 1, dtype=np.complex128)
        projected_rhs[0] = beta
        count = 0
        for column in range(restart):
            image = apply(basis[:, column])
            matvecs += 1
            image_norm = float(xp.linalg.norm(image).item())
            active = basis[:, :column + 1]
            coefficients = active.conj().T @ image
            image -= active @ coefficients
            correction = active.conj().T @ image
            image -= active @ correction
            hessenberg[:column + 1, column] = asnumpy(coefficients + correction)
            remainder = float(xp.linalg.norm(image).item())
            hessenberg[column + 1, column] = remainder
            breakdown = remainder <= np.finfo(float).eps * max(image_norm, tiny)
            if not breakdown:
                basis[:, column + 1] = image / remainder
            for previous in range(column):
                a, b = hessenberg[previous:previous + 2, column].copy()
                hessenberg[previous, column] = cosines[previous] * a + sines[previous] * b
                hessenberg[previous + 1, column] = -sines[previous].conjugate() * a + cosines[previous] * b
            a, b = hessenberg[column:column + 2, column].copy()
            magnitude = np.hypot(abs(a), abs(b))
            phase = a / abs(a) if a != 0 else 1.0 + 0.0j
            cosines[column] = abs(a) / magnitude if magnitude else 1.0
            sines[column] = phase * b.conjugate() / magnitude if magnitude else 0.0
            hessenberg[column, column] = phase * magnitude
            hessenberg[column + 1, column] = 0.0
            a, b = projected_rhs[column:column + 2].copy()
            projected_rhs[column] = cosines[column] * a + sines[column] * b
            projected_rhs[column + 1] = -sines[column].conjugate() * a + cosines[column] * b
            iterations += 1
            count = column + 1
            if abs(projected_rhs[column + 1]) <= target or breakdown:
                break
        # Small dense algebra only. Least squares handles a singular projected
        # triangular system without promoting a local failure to an exception.
        small_solution = np.linalg.lstsq(hessenberg[:count, :count],
                                        projected_rhs[:count], rcond=None)[0]
        vector += basis[:, :count] @ xp.asarray(small_solution)
        residual_norm = float(xp.linalg.norm(rhs - apply(vector)).item())
        matvecs += 1
        if residual_norm <= target:
            return vector, residual_norm / rhs_norm, iterations, matvecs, True
    return vector, residual_norm / rhs_norm, iterations, matvecs, False


def _split_center(solution, tensor, center, qn_left, qn_right, config):
    """Economy symmetry-block SVD on the active NumPy/CuPy backend."""
    if len(center) == 1:
        solution[center[0]] = tensor
        return 0.0
    left_shape = tuple(qn_left.shape[:-1])
    right_shape = tuple(qn_right.shape[:-1])
    matrix = tensor.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
    charges_left = qn_left.reshape(-1, len(solution.qntot))
    charges_right = qn_right.reshape(-1, len(solution.qntot))
    blocks = []
    all_values = []
    for charge in sorted(set(map(tuple, charges_left))):
        opposite = solution.qntot - np.asarray(charge)
        rows = np.flatnonzero(np.all(charges_left == charge, axis=1))
        columns = np.flatnonzero(np.all(charges_right == opposite, axis=1))
        if not len(columns):
            continue
        row_indices = xp.asarray(rows)
        column_indices = xp.asarray(columns)
        block = matrix[row_indices[:, None], column_indices[None, :]]
        left, values, right = xp.linalg.svd(block, full_matrices=False)
        host_values = asnumpy(values)
        offset = len(all_values)
        all_values.extend(host_values.tolist())
        blocks.append((rows, columns, left, values, right, charge, opposite, offset))
    values = np.asarray(all_values)
    total = np.linalg.norm(values)
    if total == 0 or not np.isfinite(total):
        raise FloatingPointError("local solve produced a zero or nonfinite MPS center")
    order = np.argsort(-values, kind="stable")
    rank = max(1, min(config.bond_cap, int(np.count_nonzero(values > config.cutoff * total))))
    retained = order[:rank]
    selected = {int(index): position for position, index in enumerate(retained)}
    left_new = xp.zeros((matrix.shape[0], rank), dtype=tensor.dtype)
    right_new = xp.zeros((rank, matrix.shape[1]), dtype=tensor.dtype)
    new_charges = np.empty((rank, len(solution.qntot)), dtype=qn_left.dtype)
    for rows, columns, left, singular, right, charge, opposite, offset in blocks:
        local_indices = [i for i in range(len(singular)) if offset + i in selected]
        if not local_indices:
            continue
        positions = xp.asarray([selected[offset + i] for i in local_indices])
        local_indices = xp.asarray(local_indices)
        if solution.to_right:
            left_part = left[:, local_indices]
            right_part = singular[local_indices, None] * right[local_indices, :]
            new_charges[asnumpy(positions)] = charge
        else:
            left_part = left[:, local_indices] * singular[local_indices]
            right_part = right[local_indices, :]
            new_charges[asnumpy(positions)] = opposite
        left_new[xp.asarray(rows)[:, None], positions[None, :]] = left_part
        right_new[positions[:, None], xp.asarray(columns)[None, :]] = right_part
    solution[center[0]] = left_new.reshape(left_shape + (rank,))
    solution[center[1]] = right_new.reshape((rank,) + right_shape)
    solution.qn[center[1]] = new_charges
    solution.qnidx = center[1] if solution.to_right else center[0]
    return float(np.linalg.norm(values[order[rank:]]) / total)


def _half_sweep(solution, mpo, rhs, shift, identity, environments,
                rank_one, config, statistics):
    sources = [rhs] if rank_one is None else [rhs, rank_one.u, rank_one.v]
    if solution.site_num == 1:
        centers = [[0]]
    elif solution.to_right:
        centers = [[i, i + 1] for i in range(solution.site_num - 1)]
    else:
        centers = [[i - 1, i] for i in range(solution.site_num - 1, 0, -1)]
    methods = ("System", "Enviro") if solution.to_right else ("Enviro", "System")
    for center in centers:
        qn_left, qn_right, qn_matrix = solution._get_big_qn(center)
        mask_host = get_qn_mask(qn_matrix, solution.qntot)
        mask = xp.asarray(mask_host)
        dimension = int(np.count_nonzero(mask_host))
        if dimension == 0:
            raise ValueError(f"empty local quantum-number sector at {center}")
        left = environments[0].GetLR("L", center[0] - 1, solution, mpo, method=methods[0])
        right = environments[0].GetLR("R", center[-1] + 1, solution, mpo, method=methods[1])
        expression = hop_expr(left, right, [asxp(mpo[i]) for i in center], mask.shape)
        projections = [_project(source, solution, identity, environment, center, methods)[mask]
                       for source, environment in zip(sources, environments[1:])]
        structured = xp.zeros(mask.shape, dtype=xp.complex128)

        def apply(vector):
            structured[mask] = vector
            image = complex(shift) * vector - expression(structured)[mask]
            if rank_one is not None:
                image += complex(rank_one.alpha) * projections[1] * xp.vdot(projections[2], vector)
            return image

        current = asxp(solution[center[0]])
        if len(center) == 2:
            current = xp.tensordot(current, asxp(solution[center[1]]), axes=(-1, 0))
        solved, residual, iterations, matvecs, success = _gmres(
            apply, projections[0], current[mask], config.local_restart,
            config.local_max_cycles, config.local_tolerance)
        statistics["local_solves"] += 1
        statistics["local_iterations"] += iterations
        statistics["matvecs"] += matvecs
        statistics["local_failures"] += int(not success)
        statistics["max_local_dimension"] = max(statistics["max_local_dimension"], dimension)
        statistics["max_local_residual"] = max(statistics["max_local_residual"], residual)
        if float(xp.linalg.norm(solved).item()) == 0:
            # A zero projected source can occur in a deficient basis. Keep the
            # current center and let the global certificate expose stagnation.
            statistics["local_failures"] += 1
            solved = current[mask]
        structured[mask] = solved
        statistics["discarded_norms"].append(
            _split_center(solution, structured, center, qn_left, qn_right, config))
    if solution.site_num > 1:
        solution._switch_direction()


def solve_shifted(mpo: Mpo, rhs: Mps, shift: complex, x0: Mps | None = None,
                  *, config: ShiftedSolveConfig | None = None,
                  rank_one: RankOneTerm | None = None) -> tuple[Mps, ShiftedSolveReport]:
    """Approximately solve ``(shift I - mpo + rank_one) x = rhs`` in MPS form.

    Input states and the MPO are not modified. ``x0`` is an optional warm
    start. The returned state satisfies ``bond_cap`` even when convergence is
    impossible at that cap; in that case the report explicitly says so.

    All vectors must have compatible local spaces and the same conserved
    total quantum number. For an arbitrary Liouvillian use local operator
    bases without a spurious conservation constraint. If the MPO mixes QN
    sectors, it must be built in such a full-space basis.
    """
    config = config or ShiftedSolveConfig()
    actual_backend = "gpu" if USE_GPU else "cpu"
    if config.backend not in ("auto", actual_backend):
        raise RuntimeError(f"requested {config.backend}, but Renormalizer uses {actual_backend}; "
                           "select the process backend before importing Renormalizer")
    if not np.isfinite(shift):
        raise ValueError("shift must be finite")
    if not _same_space(mpo, rhs):
        raise ValueError("MPO and right-hand side have incompatible local spaces")
    if np.any(mpo.qntot != 0):
        raise ValueError("the shifted MPO must preserve the working quantum-number sector")
    checked = [x0] if x0 is not None else []
    if rank_one is not None:
        if not np.isfinite(rank_one.alpha):
            raise ValueError("rank-one coefficient must be finite")
        checked.extend([rank_one.u, rank_one.v])
    for state in checked:
        if not _same_space(rhs, state) or not np.array_equal(rhs.qntot, state.qntot):
            raise ValueError("all shifted-system states must share local dimensions and total QN")
    started = time.perf_counter()
    source = _absorbed(rhs)
    if _norm(source) == 0:
        raise ValueError("the shifted-system right-hand side must be nonzero")
    if rank_one is not None:
        rank_one = RankOneTerm(rank_one.alpha, _absorbed(rank_one.u), _absorbed(rank_one.v))
    solution = _absorbed(x0 if x0 is not None else source)
    if _norm(solution) == 0:
        solution = source.copy()
    if x0 is None and abs(shift) > 1e-14:
        solution.scale(1 / complex(shift), inplace=True)
    solution.compress_config = CompressConfig(CompressCriteria.both,
                                              threshold=config.cutoff,
                                              max_bonddim=config.bond_cap)
    if max(solution.bond_dims) > config.bond_cap:
        solution.canonicalise().compress()
    solution.ensure_right_canonical()
    best = solution.copy()
    best_residual = shifted_residual(mpo, source, solution, shift, rank_one)
    residual_history = [best_residual]
    statistics = dict(local_solves=0, local_iterations=0, matvecs=0,
                      local_failures=0, max_local_dimension=0,
                      max_local_residual=0.0, discarded_norms=[])
    stop_reason = "converged" if best_residual <= config.tolerance else "sweep_limit"
    completed = stagnant = 0
    if best_residual > config.tolerance:
        identity = Mpo.identity(mpo.model)
        sources = [source] if rank_one is None else [source, rank_one.u, rank_one.v]
        environments = [Environ(solution, mpo, "R")]
        environments.extend(Environ(item, identity, "R", mps_conj=_ConjugatedSites(solution))
                            for item in sources)
        for sweep in range(1, config.max_sweeps + 1):
            for _ in range(1 if solution.site_num == 1 else 2):
                _half_sweep(solution, mpo, source, shift, identity, environments,
                            rank_one, config, statistics)
            completed = sweep
            residual = shifted_residual(mpo, source, solution, shift, rank_one)
            residual_history.append(residual)
            if not np.isfinite(residual):
                stop_reason = "nonfinite_residual"
                break
            stagnant = 0 if residual < (1 - config.stagnation_tolerance) * best_residual else stagnant + 1
            if residual < best_residual:
                best_residual, best = residual, solution.copy()
            if residual <= config.tolerance:
                stop_reason = "converged"
                break
            if config.stagnation_sweeps and stagnant >= config.stagnation_sweeps:
                stop_reason = "stagnation"
                break
    maximum_bond = max(map(int, best.bond_dims))
    if maximum_bond > config.bond_cap:
        raise RuntimeError("shifted solve violated its retained MPS bond cap")
    report = ShiftedSolveReport(
        converged=best_residual <= config.tolerance, residual=best_residual,
        sweeps=completed, max_bond=maximum_bond, backend=actual_backend,
        shift=complex(shift), stop_reason=stop_reason, seconds=time.perf_counter() - started,
        residual_history=residual_history, **statistics)
    return best, report
