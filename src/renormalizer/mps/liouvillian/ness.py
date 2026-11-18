"""Trace-constrained two-site NESS least squares on NumPy/CuPy.

The local rectangular map is B = L V, including output outside the MPS
frame V. QR factorizations of MPO-applied block maps give orthonormal output
frames without constructing L†L or a global dense state. Projected LSQR
solves for trace-zero corrections. Charge-block SVDs retain right-state
Schmidt information; a correction to the remaining center restores trace.

Native Renormalizer Mpo/Mps are the public interface. After initialization,
all working cores, output environments, LSQR vectors and factorizations stay
on the selected device. Only scalars, charge labels and singular values used
for rank selection cross to the host. Returned native Mps cores are on host.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np
from opt_einsum import contract

from renormalizer import Mpo, Mps
from renormalizer.mps.backend import xp, USE_GPU, backend
from renormalizer.mps.matrix import asnumpy
from .tensor import compress, inner, product_trace
from .ness_blocks import (Image, Layout, PhysicalMap, BlockLocalMap,
                          image_step, image_norm, sector_prefixes)


@dataclass(frozen=True)
class NessConfig:
    bond_cap: int = 64
    max_sweeps: int = 12
    tolerance: float = 1e-8
    trace_tolerance: float = 1e-10
    hermiticity_tolerance: float = 1e-8
    cutoff: float = 1e-12
    local_tolerance: float = 1e-11
    local_maxiter: int = 300
    validation_tolerance: float = 1e-9
    max_local_elements: int = 20_000_000
    block_workspace_elements: int = 2_000_000
    storage: str = "device"
    max_host_output_elements: int = 8_000_000_000
    max_host_coefficient_bytes: int = 128*1024**3
    host_tile_elements: int = 16_000_000
    backend: str = "auto"

    def __post_init__(self):
        for name in ("bond_cap", "max_sweeps", "local_maxiter", "max_local_elements", "block_workspace_elements",
                     "max_host_output_elements", "max_host_coefficient_bytes", "host_tile_elements"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("tolerance", "trace_tolerance", "hermiticity_tolerance",
                     "cutoff", "local_tolerance", "validation_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.cutoff >= 1:
            raise ValueError("cutoff must be below one")
        if self.backend not in ("auto", "cpu", "gpu"):
            raise ValueError("backend must be auto, cpu or gpu")
        if self.storage not in ("device", "host"):
            raise ValueError("storage must be device or host")


@dataclass
class NessResult:
    state: Mps
    converged: bool
    status: str
    residual: float
    absolute_residual: float
    trace_error: float
    hermiticity_error: float | None
    sweeps: int
    backend: str
    seconds: float
    history: list = field(default_factory=list)
    trace_preservation_error: float = 0.0

    def diagnostics(self):
        return dict(converged=self.converged, status=self.status,
                    residual=self.residual, absolute_residual=self.absolute_residual,
                    trace_error=self.trace_error, hermiticity_error=self.hermiticity_error,
                    sweeps=self.sweeps, backend=self.backend, seconds=self.seconds,
                    bond_dims=self.state.bond_dims, history=self.history,
                    trace_preservation_error=self.trace_preservation_error)


def _contract(expression, *args):
    return contract(expression, *args, optimize="greedy")


def _norm_array(array):
    return float(xp.linalg.norm(array.ravel()))


def _norm(cores):
    """Untruncated QR norm; no subtraction of nearly equal Gram scalars."""
    current = cores[0].copy()
    for following in cores[1:]:
        _, r = xp.linalg.qr(current.reshape(-1, current.shape[-1]), mode="reduced")
        current = xp.tensordot(r, following, axes=(1, 0))
    result = _norm_array(current)
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite tensor norm")
    return result


def _apply(operator, state):
    return [_contract("apqb,iqj->aipbj", w, a).reshape(
        w.shape[0]*a.shape[0], w.shape[1], w.shape[-1]*a.shape[-1])
        for w, a in zip(operator, state)]


def _difference(left, right):
    if len(left) == 1:
        return [left[0]-right[0]]
    result = [xp.concatenate((left[0], -right[0]), axis=2)]
    for a, b in zip(left[1:-1], right[1:-1]):
        c = xp.zeros((a.shape[0]+b.shape[0], a.shape[1], a.shape[2]+b.shape[2]),
                     dtype=xp.complex128)
        c[:a.shape[0], :, :a.shape[2]] = a
        c[a.shape[0]:, :, a.shape[2]:] = b
        result.append(c)
    result.append(xp.concatenate((left[-1], right[-1]), axis=0))
    return result


def _overlap_left(environment, trace, state):
    return _contract("ai,apb,ipj->bj", environment, trace.conj(), state)


def _overlap_right(environment, trace, state):
    return _contract("bj,apb,ipj->ai", environment, trace.conj(), state)


def _trace(trace, state):
    value = xp.ones((1, 1), dtype=xp.complex128)
    for t, a in zip(trace, state):
        value = _overlap_left(value, t, a)
    return complex(value[0, 0])


def _image_left(environment, operator, state):
    # Output block = orthonormal frame * environment[k, MPO bond, MPS bond].
    block = _contract("kai,apqb,iqj->kpbj", environment, operator, state)
    _, r = xp.linalg.qr(block.reshape(-1, operator.shape[-1]*state.shape[-1]),
                         mode="reduced")
    return r.reshape(-1, operator.shape[-1], state.shape[-1])


def _image_right(environment, operator, state):
    block = _contract("apqb,iqj,lbj->aipl", operator, state, environment)
    matrix = block.reshape(operator.shape[0]*state.shape[0], -1)
    _, r = xp.linalg.qr(matrix.conj().T, mode="reduced")
    return r.conj().T.reshape(operator.shape[0], state.shape[0], -1).transpose(2, 0, 1)


class _LocalMap:
    def __init__(self, left, operators, right, shape, mask, config):
        self.left, self.operators, self.right = left, operators, right
        self.shape, self.mask = shape, mask
        self.output_shape = (left.shape[0],) + tuple(w.shape[1] for w in operators) + (right.shape[0],)
        if max(math.prod(shape), math.prod(self.output_shape)) > config.max_local_elements:
            raise MemoryError("Local LSQR tensor exceeds max_local_elements; reduce bond cap or increase the explicit budget")
        self.matvecs = 0

    def forward(self, vector):
        self.matvecs += 1
        x = vector.reshape(self.shape) * self.mask
        if len(self.operators) == 1:
            return _contract("kai,apqb,lbj,iqj->kpl", self.left,
                             self.operators[0], self.right, x).ravel()
        return _contract("kai,apqc,cstb,lbj,iqtj->kpsl", self.left,
                         *self.operators, self.right, x).ravel()

    def adjoint(self, vector):
        self.matvecs += 1
        y = vector.reshape(self.output_shape)
        if len(self.operators) == 1:
            result = _contract("kai,apqb,lbj,kpl->iqj", self.left.conj(),
                               self.operators[0].conj(), self.right.conj(), y)
        else:
            result = _contract("kai,apqc,cstb,lbj,kpsl->iqtj", self.left.conj(),
                               *(w.conj() for w in self.operators), self.right.conj(), y)
        return (result*self.mask).ravel()


def _lsqr(forward, adjoint, rhs, size, tolerance, maxiter):
    """Complex Golub--Kahan LSQR, including inconsistent least squares."""
    x = xp.zeros(size, dtype=xp.complex128)
    beta = _norm_array(rhs)
    if beta == 0:
        return x, 0
    u = rhs/beta
    v = adjoint(u)
    alpha = _norm_array(v)
    if alpha == 0:
        return x, 0
    v /= alpha
    w = v.copy()
    rhobar, phibar, bnorm, anorm2 = alpha, beta, beta, alpha*alpha
    for iteration in range(1, maxiter+1):
        u = forward(v)-alpha*u
        beta = _norm_array(u)
        if beta:
            u /= beta
        v = adjoint(u)-beta*v
        alpha = _norm_array(v)
        if alpha:
            v /= alpha
        anorm2 += beta*beta + alpha*alpha
        rho = math.hypot(rhobar, beta)
        if rho == 0:
            break
        c, s = rhobar/rho, beta/rho
        theta, rhobar = s*alpha, -c*alpha
        phi, phibar = c*phibar, s*phibar
        x += (phi/rho)*w
        w = v-(theta/rho)*w
        arnorm = alpha*abs(s*phi)
        if (abs(phibar) <= tolerance*bnorm or
                arnorm <= tolerance*math.sqrt(anorm2)*abs(phibar) or alpha == 0):
            break
    return x, iteration


def _split(theta, row_charges, column_charges, to_right, config):
    """Charge-block SVD. Charges use the prefix convention on both sides."""
    matrix = theta.reshape(len(row_charges), len(column_charges))
    blocks, values = [], []
    for charge in sorted(set(map(tuple, row_charges))):
        rows = np.flatnonzero(np.all(row_charges == charge, axis=1))
        cols = np.flatnonzero(np.all(column_charges == charge, axis=1))
        if not len(cols):
            continue
        ri, ci = xp.asarray(rows), xp.asarray(cols)
        u, s, vh = xp.linalg.svd(matrix[ri[:, None], ci[None, :]], full_matrices=False)
        blocks.append((ri, ci, u, s, vh, charge, len(values)))
        values.extend(asnumpy(s).tolist())
    values = np.asarray(values)
    total = float(np.linalg.norm(values))
    if not total or not np.isfinite(total):
        raise FloatingPointError("Local solve lost the normalized state")
    order = np.argsort(-values, kind="stable")
    rank = max(1, min(config.bond_cap, int(np.count_nonzero(values > config.cutoff*total))))
    positions = {int(old): new for new, old in enumerate(order[:rank])}
    left = xp.zeros((matrix.shape[0], rank), dtype=xp.complex128)
    right = xp.zeros((rank, matrix.shape[1]), dtype=xp.complex128)
    labels = np.empty((rank, row_charges.shape[1]), dtype=int)
    for rows, cols, u, s, vh, charge, offset in blocks:
        selected = [k for k in range(len(s)) if offset+k in positions]
        if not selected:
            continue
        dst = xp.asarray([positions[offset+k] for k in selected])
        src = xp.asarray(selected)
        left[rows[:, None], dst[None, :]] = u[:, src]*(1 if to_right else s[src])
        right[dst[:, None], cols[None, :]] = vh[src, :]*(s[src, None] if to_right else 1)
        labels[asnumpy(dst)] = charge
    discarded = float(np.linalg.norm(values[order[rank:]])/total)
    return left.reshape(theta.shape[:2]+(rank,)), right.reshape((rank,)+theta.shape[2:]), labels, discarded


def _restore_trace(left, right, functional, to_right, left_mask, right_mask):
    # functional is a linear covector, WITHOUT complex conjugation of theta.
    if to_right:
        covector = _contract("ipqj,ipk->kqj", functional, left)*right_mask
        center = right
    else:
        covector = _contract("ipqj,kqj->ipk", functional, right)*left_mask
        center = left
    denominator = float(xp.vdot(covector, covector).real)
    if denominator < np.finfo(float).tiny:
        raise FloatingPointError("Compression removed all trace support; increase bond_cap")
    error = 1-complex(xp.sum(covector*center))
    correction = covector.conj()*(error/denominator)
    center += correction
    return _norm_array(correction)


def _half_sweep(state, charges, physical, operator, trace, to_right, config,
                mpo_charges, allowed, physical_maps, local_callback=None):
    n = len(state)
    def boundary(right):
        return Image(xp.ones((1, 1, 1), dtype=xp.complex128),
                     charges[-1].copy() if right else charges[0].copy())
    def step(environment, j, direction):
        bond = j+1 if direction else j
        return image_step(environment, operator[j], state[j], charges[bond], mpo_charges[bond],
                          physical[j], allowed[bond], direction)
    overlap_boundary = xp.ones((1, 1), dtype=xp.complex128)
    images, overlaps = [None]*(n+1), [None]*(n+1)
    if to_right:
        images[n], overlaps[n] = boundary(True), overlap_boundary
        for j in range(n-1, 1, -1):
            images[j] = step(images[j+1], j, False)
            overlaps[j] = _overlap_right(overlaps[j+1], trace[j], state[j])
        moving_image, moving_overlap = boundary(False), overlap_boundary
        sites = range(max(1, n-1))
    else:
        images[0], overlaps[0] = boundary(False), overlap_boundary
        for j in range(max(0, n-2)):
            images[j+1] = step(images[j], j, True)
            overlaps[j+1] = _overlap_left(overlaps[j], trace[j], state[j])
        moving_image, moving_overlap = boundary(True), overlap_boundary
        sites = range(n-2, -1, -1) if n > 1 else [0]
    statistics = dict(local_iterations=0, matvecs=0, discarded=0., trace_correction=0.,
                      max_packed_input=0, max_packed_output=0, max_coefficient_bytes=0)
    for j in sites:
        local_started = time.perf_counter()
        end = min(j+2, n)
        left_image, right_image = (moving_image, images[end]) if to_right else (images[j], moving_image)
        lt, rt = (moving_overlap, overlaps[end]) if to_right else (overlaps[j], moving_overlap)
        if end-j == 1:
            theta = state[j].copy()
            functional = _contract("ai,apb,bj->ipj", lt, trace[j].conj(), rt)
        else:
            theta = xp.tensordot(state[j], state[j+1], axes=(2, 0))
            functional = _contract("ai,apc,cqb,bj->ipqj", lt, trace[j].conj(), trace[j+1].conj(), rt)
            row = (charges[j][:, None, :]+physical[j][None, :, :]).reshape(-1, physical[j].shape[1])
            col = (charges[end][None, :, :]-physical[j+1][:, None, :]).reshape(-1, physical[j].shape[1])
        layout = Layout(charges[j], physical[j:end], charges[end])
        output_layout = Layout(left_image.charges, physical[j:end], right_image.charges)
        action = BlockLocalMap(left_image, physical_maps[j], right_image, layout, output_layout,
                               mpo_charges[j], mpo_charges[end], config)
        t = layout.pack(functional.conj())
        tnorm = float(xp.vdot(t, t).real)
        if tnorm < np.finfo(float).tiny:
            raise FloatingPointError("Local frame cannot represent nonzero trace")
        a = layout.pack(theta)
        a += t*((1-xp.vdot(t, a))/tnorm)
        def project(v):
            return v-t*(xp.vdot(t, v)/tnorm)
        setup_seconds = time.perf_counter()-local_started
        initial_residual = action.forward(a)
        delta, iterations = _lsqr(lambda v: action.forward(project(v)),
                                  lambda v: project(action.adjoint(v)), -initial_residual,
                                  a.size, config.local_tolerance, config.local_maxiter)
        candidate = a+project(delta)
        # Finite-precision LSQR must improve the actual rectangular objective.
        if _norm_array(action.forward(candidate)) <= _norm_array(initial_residual)*(1+1e-12):
            a = candidate
        a += t*((1-xp.vdot(t, a))/tnorm)
        if end-j == 1:
            state[j] = layout.unpack(a)
        else:
            left, right, labels, discarded = _split(layout.unpack(a), row, col, to_right, config)
            left_mask = xp.asarray(np.all(charges[j][:, None, None, :]+physical[j][None, :, None, :]
                                         == labels[None, None, :, :], axis=-1))
            right_mask = xp.asarray(np.all(labels[:, None, None, :]+physical[j+1][None, :, None, :]
                                          == charges[end][None, None, :, :], axis=-1))
            correction = _restore_trace(left, right, functional, to_right, left_mask, right_mask)
            state[j], state[j+1], charges[j+1] = left, right, labels
            statistics["discarded"] = max(statistics["discarded"], discarded)
            statistics["trace_correction"] = max(statistics["trace_correction"], correction)
        statistics["local_iterations"] += iterations
        statistics["matvecs"] += action.matvecs
        for key, value in (("max_packed_input", layout.size), ("max_packed_output", output_layout.size),
                           ("max_coefficient_bytes", action.coefficient_bytes)):
            statistics[key] = max(statistics[key], value)
        if local_callback:
            local_callback(dict(site=j, to_right=to_right, iterations=iterations,
                                packed_input=layout.size, packed_output=output_layout.size,
                                dense_output=math.prod(output_layout.shape),
                                blocks=action.block_count, batches=len(action.batches),
                                coefficient_bytes=action.coefficient_bytes,
                                setup_seconds=setup_seconds, seconds=time.perf_counter()-local_started,
                                max_bond=max(a.shape[-1] for a in state)))
        del action, initial_residual, candidate, delta, a, theta, functional, t
        images[end if to_right else j] = None
        overlaps[end if to_right else j] = None
        index = j if to_right else end-1
        if to_right:
            moving_image = step(moving_image, index, True)
            moving_overlap = _overlap_left(moving_overlap, trace[index], state[index])
        else:
            moving_image = step(moving_image, index, False)
            moving_overlap = _overlap_right(moving_overlap, trace[index], state[index])
    return statistics


def solve_ness(liouvillian: Mpo, initial: Mps | None = None, *, trace="paired",
               config: NessConfig | None = None, paired: bool = True, callback=None,
               local_callback=None, checkpoint_callback=None) -> NessResult:
    """Minimize ||L rho|| with exact trace and charge constraints.

    ``trace='paired'`` asserts local p=ket*d+bra. A fixed-number sector MUST
    supply its sector-identity Mps; the initial/reference state is normalized
    from that identity if omitted. ``paired`` enables a physical Hermiticity
    diagnostic using the same local layout (False for other operator bases).
    Unique NESS in the selected sector is assumed, not proved. Positivity is
    not imposed. Tolerance is ||L rho||_F/||rho||_F, in generator rate units.
    """
    cfg = config or NessConfig()
    if not isinstance(cfg, NessConfig):
        raise TypeError("config must be a NessConfig")
    actual_backend = "gpu" if USE_GPU else "cpu"
    if cfg.backend not in ("auto", actual_backend):
        raise RuntimeError(f"Requested {cfg.backend}, but Renormalizer initialized {actual_backend}")
    if backend.is_32bits:
        raise ValueError("NESS validation requires Renormalizer float64/complex128")
    if not isinstance(liouvillian, Mpo) or liouvillian.site_num < 1:
        raise TypeError("liouvillian must be a nonempty native Mpo")
    if isinstance(trace, str):
        if trace != "paired":
            raise ValueError("trace must be 'paired' or an explicit Mps")
        trace = product_trace(liouvillian.model)
    if not isinstance(trace, Mps):
        raise TypeError("trace must be an Mps")
    for item in (trace, initial if initial is not None else trace):
        if item.site_num != liouvillian.site_num or list(item.pbond_list) != list(liouvillian.pbond_list):
            raise ValueError("MPO, trace and initial local dimensions must match")
        if not np.array_equal(item.qntot, trace.qntot):
            raise ValueError("Trace and initial state must have the same total charges")
        if any(not np.array_equal(a.sigmaqn, b.sigmaqn)
               for a, b in zip(item.model.basis, liouvillian.model.basis)):
            raise ValueError("MPO, trace and initial local charge bases must match")
    if np.any(liouvillian.qntot):
        raise ValueError("Liouvillian must conserve the selected total charges")
    dimensions = [math.isqrt(int(p)) for p in trace.pbond_list]
    if paired and any(d*d != p for d, p in zip(dimensions, trace.pbond_list)):
        raise ValueError("paired=True requires square local operator dimensions")
    if cfg.storage == "host":
        from .ness_host import solve_host
        return solve_host(liouvillian, initial, trace, cfg, paired, callback,
                          local_callback, checkpoint_callback)
    started = time.perf_counter()
    guess = compress(initial if initial is not None else trace, cfg.bond_cap, cfg.cutoff)
    amplitude = inner(trace, guess)
    if abs(amplitude) < np.finfo(float).tiny:
        raise ValueError("Initial/compressed state has zero trace")
    guess = guess.scale(1/amplitude)
    guess.ensure_right_canonical()
    guess.move_qnidx(guess.site_num)  # export all bond charges as prefixes
    charges = [q.copy() for q in guess.qn]
    physical = [np.asarray(b.sigmaqn) for b in guess.model.basis]
    state = [xp.asarray(a.array, dtype=xp.complex128) for a in guess]
    state[0] *= complex(guess.coeff)
    trace_cores = [xp.asarray(a.array, dtype=xp.complex128) for a in trace]
    trace_cores[0] = trace_cores[0]*complex(trace.coeff)
    operator = [xp.asarray(a.array, dtype=xp.complex128) for a in liouvillian]
    prefix_mpo = liouvillian.copy()
    prefix_mpo.move_qnidx(prefix_mpo.site_num)
    mpo_charges = [q.copy() for q in prefix_mpo.qn]
    allowed = sector_prefixes(physical, guess.qntot)
    physical_maps = [PhysicalMap([liouvillian[k].array for k in range(j, min(j+2, len(state)))])
                     for j in range(max(1, len(state)-1))]
    adjoint = [w.conj().swapaxes(1, 2) for w in operator]
    trace_defect = _norm(_apply(adjoint, trace_cores))/_norm(trace_cores)
    if trace_defect > cfg.validation_tolerance:
        raise ValueError(f"L† trace is nonzero: {trace_defect:.3e}; check sector/vectorization")

    def audit(sweep, stats=None):
        size = _norm(state)
        absolute = image_norm(operator, state, charges, mpo_charges, physical, allowed)
        if not np.isfinite(absolute):
            raise FloatingPointError("Nonfinite full Liouvillian residual")
        herm = None
        if paired:
            dagger = [a.reshape(a.shape[0], d, d, a.shape[2]).swapaxes(1, 2).conj().reshape(a.shape)
                      for a, d in zip(state, dimensions)]
            herm = _norm(_difference(state, dagger))/size
        record = dict(sweep=sweep, residual=absolute/size, absolute_residual=absolute,
                      trace_error=abs(_trace(trace_cores, state)-1), hermiticity_error=herm,
                      norm=size, max_bond=max(a.shape[-1] for a in state), seconds=time.perf_counter()-started)
        if stats:
            record.update(local_iterations=sum(s["local_iterations"] for s in stats),
                          matvecs=sum(s["matvecs"] for s in stats),
                          discarded=max(s["discarded"] for s in stats),
                          trace_correction=max(s["trace_correction"] for s in stats))
            for key in ("max_packed_input", "max_packed_output", "max_coefficient_bytes"):
                record[key] = max(s[key] for s in stats)
        if callback:
            callback(dict(record))
        return record

    def converged(record):
        return (record["residual"] <= cfg.tolerance and record["trace_error"] <= cfg.trace_tolerance
                and (record["hermiticity_error"] is None or record["hermiticity_error"] <= cfg.hermiticity_tolerance))

    history = [audit(0)]
    best = history[0]
    best_state, best_charges = [a.copy() for a in state], [q.copy() for q in charges]
    def export(cores, labels):
        output = Mps.from_mp(guess.model, [asnumpy(a) for a in cores])
        output.qn, output.qntot = [q.copy() for q in labels], guess.qntot.copy()
        output.qnidx = len(state)
        output.move_qnidx(0)
        output.to_right, output.coeff = True, 1.0
        return output
    for sweep in range(1, cfg.max_sweeps+1):
        if converged(best):
            break
        stats = [_half_sweep(state, charges, physical, operator, trace_cores, direction, cfg,
                             mpo_charges, allowed, physical_maps, local_callback)
                 for direction in (True, False)]
        record = audit(sweep, stats)
        history.append(record)
        if record["residual"] < best["residual"] or converged(record):
            best = record
            best_state, best_charges = [a.copy() for a in state], [q.copy() for q in charges]
        if checkpoint_callback:
            checkpoint_callback(export(best_state, best_charges), dict(best))
    output = export(best_state, best_charges)
    ok = converged(best)
    return NessResult(output, ok, "converged" if ok else "sweep_budget", best["residual"],
                      best["absolute_residual"], best["trace_error"], best["hermiticity_error"],
                      len(history)-1, actual_backend, time.perf_counter()-started, history, trace_defect)
