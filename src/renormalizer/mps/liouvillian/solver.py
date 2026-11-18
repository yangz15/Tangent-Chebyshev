"""Low-rank Lyapunov discovery and original-MPO Schur extraction.

Only dictionaries, covariance-factor Gram matrices and projected eigensystems
are dense. All full-space vectors, including ADI residual factors, inverse
solutions, restarts and returned eigenoperators, are truncated native MPS.
There is no dense Hilbert-space fallback. See the package README for the
distinction between residual convergence and rightmost-spectrum coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Sequence, Callable
import time

import numpy as np
import scipy.linalg as la

from renormalizer.mps import Mps, Mpo
from renormalizer.mps.backend import USE_GPU, backend, xp
from .tensor import inner, norm, compress, combine, expectation, eigen_residual, product_trace
from .shifted import ShiftedSolveConfig, RankOneTerm, solve_shifted


@dataclass
class SpectrumConfig:
    """Numerical budgets; tolerances refer to original-operator residuals.

    ``poles`` are right-half-plane ADI z values, not the negative p convention.
    ``frequency_window`` supplies exploration coverage, never exact eigenvalues.
    ``stability_shift`` shifts the auxiliary operator only; returned eigenvalues
    always come from the original MPO. It must exceed any positive real part
    if an input outside the Lindblad/stable-sector contract is supplied.
    """
    bond_cap: int = 32
    cutoff: float = 1e-12
    covariance_rank: int = 4
    max_subspace: int = 40
    max_outer: int = 12
    shifts_per_outer: int = 4
    poles: Optional[Sequence[complex]] = None
    damping: float = 0.1
    frequency_window: Optional[tuple[float, float]] = None
    stability_shift: float = 0.01
    tolerance: float = 1e-7
    lyapunov_tolerance: float = 0.1
    gram_cutoff: float = 1e-12
    guard: int = 2
    refinement_steps: int = 3
    restart_tolerance: float = 1e-6
    seed: int = 20260914
    min_outer: int = 2
    max_seconds: Optional[float] = None
    ness_scale: float = 1.0
    inner: ShiftedSolveConfig = field(default_factory=ShiftedSolveConfig)

    def __post_init__(self):
        for key in ('bond_cap', 'covariance_rank', 'max_subspace', 'max_outer',
                    'shifts_per_outer', 'min_outer'):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('guard', 'refinement_steps'):
            if isinstance(getattr(self, key), bool) or not isinstance(getattr(self, key), int) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        for key in ('cutoff', 'damping', 'stability_shift', 'tolerance',
                    'lyapunov_tolerance', 'gram_cutoff', 'restart_tolerance', 'ness_scale'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.cutoff >= 1 or self.gram_cutoff >= 1:
            raise ValueError('cutoff and gram_cutoff must be below one')
        if self.covariance_rank > self.max_subspace:
            raise ValueError('covariance_rank cannot exceed max_subspace')
        if not isinstance(self.inner, ShiftedSolveConfig):
            raise TypeError('inner must be a ShiftedSolveConfig')
        if self.frequency_window is not None:
            lo, hi = self.frequency_window
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError('frequency_window must be a finite increasing pair')
        if self.poles is not None:
            if len(self.poles) == 0 or any(not np.isfinite(z) or complex(z).real <= 0 for z in self.poles):
                raise ValueError('ADI poles must be finite and in the right half-plane')
        if self.max_seconds is not None and (not np.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError('max_seconds must be positive')


@dataclass
class SpectrumResult:
    """Physical capped MPS eigenoperators and diagnostics, never dense states.

    ``converged`` certifies residual/trace tests for the requested count. It is
    not a mathematical certificate that no unexcited rightmost mode was missed.
    ``coverage_certified`` is deliberately false without an external enclosure.
    """
    eigenvalues: np.ndarray
    eigenstates: list
    residuals: np.ndarray
    status: str
    converged: bool
    ness: Optional[Mps] = None
    ness_residual: Optional[float] = None
    ness_trace_error: Optional[float] = None
    history: list = field(default_factory=list)
    backend: str = 'cpu'
    seconds: float = 0.0
    max_bond: int = 0
    coverage_certified: bool = False
    scope: str = 'seeded invariant sector'

    @property
    def all_eigenvalues(self):
        """NESS eigenvalue zero followed by the requested decay eigenvalues."""
        return np.concatenate(([0j], self.eigenvalues)) if self.ness is not None else self.eigenvalues.copy()

    @property
    def all_eigenstates(self):
        """Native MPS in the same order as :attr:`all_eigenvalues`."""
        return ([self.ness] if self.ness is not None else []) + list(self.eigenstates)

    @property
    def all_residuals(self):
        """Original-MPO relative residuals in the combined eigenvalue order."""
        return np.concatenate(([self.ness_residual], self.residuals)) if self.ness is not None else self.residuals.copy()

    def diagnostics(self):
        """JSON-serializable metadata without copying or densifying MPS data."""
        def serial(value):
            if isinstance(value, complex):
                return dict(real=value.real, imag=value.imag)
            if isinstance(value, np.ndarray):
                return serial(value.tolist())
            if isinstance(value, np.generic):
                return serial(value.item())
            if isinstance(value, dict):
                return {key: serial(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serial(item) for item in value]
            return value
        return serial(dict(eigenvalues=self.all_eigenvalues, residuals=self.all_residuals,
                           status=self.status, converged=self.converged,
                           backend=self.backend, seconds=self.seconds,
                           max_bond=self.max_bond, scope=self.scope,
                           coverage_certified=self.coverage_certified,
                           ness_trace_error=self.ness_trace_error,
                           history=self.history))


def _normalized(x, cfg):
    x = compress(x, cfg.bond_cap, cfg.cutoff)
    size = norm(x)
    if not np.isfinite(size) or size < 1e-14:
        return None
    return x.scale(1 / size)


class _Projector:
    """Orthogonal finite-rank projection, with independently capped outputs."""
    def __init__(self, states, cfg):
        self.states, self.cfg = list(states), cfg
        if self.states:
            g = np.array([[inner(x, y) for y in self.states] for x in self.states])
            vals, vecs = la.eigh((g + g.conj().T) / 2)
            keep = vals > max(vals[-1] * cfg.gram_cutoff, 1e-15)
            self.inverse = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].conj().T
        else:
            self.inverse = np.zeros((0, 0), complex)

    def apply(self, x, exact=False):
        if not self.states:
            return x.copy() if exact else compress(x, self.cfg.bond_cap, self.cfg.cutoff)
        coeff = self.inverse @ np.array([inner(q, x) for q in self.states])
        return combine([x] + self.states, [1] + list(-coeff),
                       None if exact else self.cfg.bond_cap, self.cfg.cutoff)

    def loss(self, x):
        if not self.states:
            return 0.0
        c = np.array([inner(q, x) for q in self.states])
        return float(np.sqrt(max(0., (c.conj() @ self.inverse @ c).real)))


class _Space:
    def __init__(self, mpo, cfg, trace):
        self.mpo, self.cfg, self.trace = mpo, cfg, trace
        self.locked = []
        self.solve_count = 0
        self.rebuild()

    def rebuild(self):
        self.projector = _Projector(([self.trace] if self.trace is not None else []) + self.locked, self.cfg)

    def action(self, x, exact=False):
        # P(L-sI)P represents the quotient on range(P); the ambient zeros are
        # excluded by projecting all source/solution vectors, never eigensolved.
        px = self.projector.apply(x, exact=True)
        ax = self.mpo.apply(px)
        y = combine([ax, px], [1, -self.cfg.stability_shift],
                    None if exact else self.cfg.bond_cap, self.cfg.cutoff)
        return self.projector.apply(y, exact=exact)

    def solve(self, rhs, z, x0=None):
        rhs = self.projector.apply(rhs)
        rhs_size = norm(rhs)
        if rhs_size < 1e-14:
            return None, {'converged': True, 'residual': 0., 'zero_source': True}
        self.solve_count += 1
        config = replace(self.cfg.inner, bond_cap=self.cfg.bond_cap, cutoff=self.cfg.cutoff)
        # For an invariant locked block, P(z+s-L)^(-1)P is exactly the
        # resolvent of the complementary compression. Approximate locking and
        # MPS truncation are checked by the compressed-operator residual below.
        x, report = solve_shifted(self.mpo, rhs, z + self.cfg.stability_shift,
                                 x0=x0, config=config)
        x = self.projector.apply(x)
        ax = self.action(x, exact=True)
        residual = norm(combine([rhs, x, ax], [1, -z, 1], None, self.cfg.cutoff)) / rhs_size
        record = asdict(report)
        record.update(shift=complex(z), projected_residual=float(residual),
                      projected_leakage=self.projector.loss(x),
                      converged=bool(residual <= config.tolerance), max_bond=max(x.bond_dims))
        return x, record


class _Dictionary:
    """Incremental original-MPO projection; no rank-expanded images are cached."""
    def __init__(self, mpo, cfg, trace=None):
        self.mpo, self.cfg, self.trace = mpo, cfg, trace
        self.states = []
        self.g = np.zeros((0, 0), complex)
        self.h = np.zeros((0, 0), complex)

    def append(self, x):
        x = _normalized(x, self.cfg)
        if x is None:
            return False
        n = len(self.states)
        c = np.array([inner(v, x) for v in self.states])
        if n:
            w = self.whiten()
            novelty = 1 - float(la.norm(w.conj().T @ c)**2)
            if novelty <= self.cfg.gram_cutoff * 0.1:
                return False
        if n >= self.cfg.max_subspace:
            return False
        g, h = np.zeros((n+1, n+1), complex), np.zeros((n+1, n+1), complex)
        g[:n, :n], h[:n, :n] = self.g, self.h
        g[:n, n], g[n, :n] = c, c.conj()
        g[n, n] = inner(x, x)
        for i, old in enumerate(self.states):
            h[i, n] = expectation(self.mpo, old, x)
            h[n, i] = expectation(self.mpo, x, old)
        h[n, n] = expectation(self.mpo, x, x)
        self.states.append(x)
        self.g, self.h = g, h
        return True

    def whiten(self):
        if not self.states:
            raise ValueError('No nonzero independent search directions')
        s, u = la.eigh((self.g + self.g.conj().T) / 2)
        keep = s > max(s[-1] * self.cfg.gram_cutoff, 1e-15)
        if not np.any(keep):
            raise ValueError('Numerically empty dictionary')
        return u[:, keep] / np.sqrt(s[keep])

    def materialize(self, c):
        return combine(self.states, c, self.cfg.bond_cap, self.cfg.cutoff)

    def extract(self, count):
        w = self.whiten()
        h = w.conj().T @ self.h @ w
        # Schur factorization is deliberately confined to this small matrix.
        t, u = la.schur(h, output='complex')
        vals, coeff = la.eig(t)
        order = np.lexsort((vals.imag, -vals.real))
        c = w @ u @ coeff[:, order]
        vals = vals[order]
        out, errors, accepted_vals, coefficients = [], [], [], []
        for value, col in zip(vals, c.T):
            x = _normalized(self.materialize(col), self.cfg)
            if x is None:
                continue
            if self.trace is not None:
                # Returned decay modes must remain in the represented trace-free
                # space after their final materialization and truncation.
                x = _Projector([self.trace], self.cfg).apply(x)
                x = _normalized(x, self.cfg)
                if x is None:
                    continue
            error = eigen_residual(self.mpo, x, value)
            out.append(x)
            errors.append(error)
            accepted_vals.append(value)
            coefficients.append(col)
            if len(out) >= count:
                break
        return dict(values=np.asarray(accepted_vals, complex), states=out,
                    residuals=np.asarray(errors), coefficients=np.asarray(coefficients).T,
                    reduced_dimension=h.shape[0])


def _subspace_loss(states, target, cutoff):
    """Largest lost principal component, using only physical MPS overlaps."""
    if not target:
        return 0.0
    if not states:
        return 1.0
    g = np.array([[inner(a, b) for b in states] for a in states])
    gt = np.array([[inner(a, b) for b in target] for a in target])
    cross = np.array([[inner(a, b) for b in target] for a in states])
    s, u = la.eigh((g+g.conj().T)/2)
    st, ut = la.eigh((gt+gt.conj().T)/2)
    keep, keept = s > max(s[-1]*cutoff, 1e-15), st > max(st[-1]*cutoff, 1e-15)
    if np.count_nonzero(keept) < len(target):
        return 1.0
    q = u[:, keep]/np.sqrt(s[keep])
    qt = ut[:, keept]/np.sqrt(st[keept])
    singular = la.svdvals(q.conj().T@cross@qt)
    if len(singular) < len(target):
        return 1.0
    return float(np.sqrt(max(0., 1-float(singular[-1])**2)))


def _factor_compress(states, rank, cfg):
    """PSD covariance truncation using physical factor overlaps, then MPS SVD."""
    if not states:
        return [], {'rank': 0, 'tail': 1.0, 'factor_truncation': []}
    gram = np.array([[inner(a, b) for b in states] for a in states])
    vals, u = la.eigh((gram + gram.conj().T) / 2)
    order = np.argsort(-vals)
    vals, u = np.maximum(vals[order], 0), u[:, order]
    keep = min(rank, np.count_nonzero(vals > max(vals[0]*cfg.gram_cutoff, 1e-28)))
    result, errors = [], []
    for col in u[:, :keep].T:
        x = combine(states, col, cfg.bond_cap, cfg.cutoff)
        # Exact transient tensor difference measures loss in materializing each
        # covariance factor; it is not hidden in the small covariance tail.
        exact = combine(states, col, None, cfg.cutoff)
        errors.append(norm(combine([exact, x], [1, -1], None, cfg.cutoff)))
        result.append(x)
    total = np.sqrt(sum(norm(x)**2 for x in result))
    if total > 1e-14:
        result = [x.scale(1/total) for x in result]
    else:
        result = []
    return result, dict(rank=keep, tail=float(la.norm(vals[keep:])/max(la.norm(vals), 1e-300)),
                        covariance_weights=vals.tolist(), factor_truncation=errors)


def _lyapunov_residual(space, z, b):
    """True residual of represented capped factors, using only tensor overlaps.

    Uses the Gram identity at discovery accuracy. Reports a roundoff floor
    rather than declaring cancellation-dominated values to be exact zeros.
    """
    if not z:
        return 1.0, 0.0
    az = [space.action(x, exact=True) for x in z]
    f = az + z + b
    g = np.array([[inner(a, c) for c in f] for a in f])
    k, p = len(z), len(b)
    j = np.zeros((2*k+p, 2*k+p), complex)
    j[:k, k:2*k] = j[k:2*k, :k] = np.eye(k)
    j[2*k:, 2*k:] = np.eye(p)
    terms = (j @ g) * (j @ g).T
    value = float(np.sum(terms).real)
    floor2 = 100 * np.finfo(float).eps * float(np.sum(np.abs(terms)))
    denominator = la.norm(g[2*k:, 2*k:], 'fro')
    return float(np.sqrt(max(value, floor2, 0))/max(denominator, 1e-300)), float(np.sqrt(floor2)/max(denominator, 1e-300))


def _poles(cfg, values, outer):
    if cfg.poles is not None:
        return [complex(cfg.poles[(outer*cfg.shifts_per_outer+j) % len(cfg.poles)])
                for j in range(cfg.shifts_per_outer)]
    values = np.asarray(values)
    result = []
    # Leave at least one exploratory shift; never let an unstable Ritz real
    # part cross the ADI shift into the left half-plane.
    for value in values[:max(0, cfg.shifts_per_outer-1)]:
        a = max(cfg.damping * 0.05, cfg.stability_shift - value.real)
        result.append(complex(a, value.imag))
    if cfg.frequency_window is None:
        width = max(cfg.damping, float(np.max(np.abs(values.imag))) if len(values) else cfg.damping)
        lo, hi = -2*width, 2*width
    else:
        lo, hi = cfg.frequency_window
    while len(result) < cfg.shifts_per_outer:
        phase = ((outer*cfg.shifts_per_outer+len(result)+1)*0.6180339887498949) % 1
        result.append(complex(cfg.damping, lo+(hi-lo)*phase))
    return result


def _make_seeds(mpo, seeds, cfg, count):
    if isinstance(seeds, Mps):
        seeds = [seeds]
    if seeds is not None:
        seeds = list(seeds)
        if not seeds:
            raise ValueError('seeds must be nonempty')
        return seeds
    # Reproducible convenience initializer for qntot=0. Nontrivial sectors
    # should supply native charged seeds instead of guessing their total QN.
    saved = np.random.get_state()
    saved_gpu = xp.random.get_random_state() if USE_GPU else None
    np.random.seed(cfg.seed)
    if USE_GPU:
        xp.random.seed(cfg.seed)
    try:
        return [Mps.random(mpo.model, np.zeros(mpo.model.qn_size, int),
                           min(cfg.bond_cap, 4)).to_complex() for _ in range(count)]
    finally:
        np.random.set_state(saved)
        if USE_GPU:
            xp.random.set_random_state(saved_gpu)


def solve_liouvillian(mpo: Mpo, seeds=None, *, n_modes: int = 2,
                      trace=None, compute_ness: bool = False,
                      config: Optional[SpectrumConfig] = None,
                      callback: Optional[Callable[[dict], None]] = None) -> SpectrumResult:
    """Compute rightmost Liouvillian modes using native truncatable MPS.

    Parameters
    ----------
    mpo : Mpo
        Original generator; never converted to a global dense/sparse matrix.
    seeds : Mps or sequence of Mps, optional
        Independent guesses in one common QN sector. Defaults to native random
        qntot=0 MPS. Repeat over all relevant sectors for a global gap claim.
    n_modes : int
        Number of decay modes when ``trace`` is supplied; otherwise number of
        rightmost modes of the supplied operator/sector, including zero modes.
    trace : Mps, 'paired', or None
        Vector implementing Tr(x)=<trace|x>. 'paired' constructs the identity
        for explicit site-local |ket,bra> vectorization. None makes no implicit
        assumption about the MPO's vectorization or its stationary directions.
    compute_ness : bool
        Also solve (-L+alpha|b><trace|)rho=alpha*b; requires a trace functional
        and a unique NESS in its declared sector. Normalize the result to trace
        one and independently certify L*rho after truncation.
    config : SpectrumConfig
        Explicit tensor, subspace, inner-solver, accuracy and work budgets.
    callback : callable, optional
        Receives each completed outer-iteration record. Exceptions propagate.

    Notes
    -----
    GPU is selected before importing Renormalizer (RENO_REQUIRE_GPU=1).
    Small dense projections run on the host; local tensor contractions and
    GMRES use the native NumPy/CuPy backend. Residual convergence is not a
    completeness certificate for a nonnormal spectrum or incomplete seeds.
    """
    cfg = config or SpectrumConfig()
    requested_backend = cfg.inner.backend
    actual_backend = 'gpu' if USE_GPU else 'cpu'
    if requested_backend != 'auto' and requested_backend != actual_backend:
        raise RuntimeError(f'Requested {requested_backend} backend but Renormalizer initialized {actual_backend}')
    if not isinstance(mpo, Mpo):
        raise TypeError('mpo must be a native Renormalizer Mpo')
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 0:
        raise ValueError('n_modes must be a nonnegative integer')
    if n_modes == 0 and not compute_ness:
        raise ValueError('Request a NESS or at least one mode')
    if cfg.max_subspace < n_modes + cfg.guard + 1:
        raise ValueError('max_subspace must leave room for targets, guards and enrichment')
    if isinstance(trace, str):
        if trace != 'paired':
            raise ValueError("trace string must be 'paired'")
        trace = product_trace(mpo.model)
    if trace is not None and not isinstance(trace, Mps):
        raise TypeError("trace must be an Mps, 'paired', or None")
    if trace is not None and tuple(trace.pbond_list) != tuple(mpo.model.pbond_list):
        raise ValueError('Trace MPS and MPO physical dimensions differ')
    if compute_ness and trace is None:
        raise ValueError("NESS normalization requires trace=Mps or trace='paired'")
    started = time.perf_counter()
    history = []
    rho, ness_error, trace_error = None, None, None
    if trace is not None:
        if norm(trace) == 0:
            raise ValueError('trace functional is zero')
        # Trace preservation is tested directly with L†, never inferred from
        # local physical dimensions or the caller naming the object Liouvillian.
        trace_defect = norm(mpo.conj_trans().apply(trace)) / norm(trace)
        if trace_defect > max(cfg.tolerance*10, 1e-10):
            raise ValueError(f'Provided trace is not preserved: ||L†trace||/||trace||={trace_defect:g}')
    if compute_ness:
        b = compress(trace.scale(1 / inner(trace, trace)), cfg.bond_cap, cfg.cutoff)
        btrace = inner(trace, b)
        if abs(btrace) < 1e-14:
            raise ValueError('The capped NESS reference has zero trace; increase bond_cap')
        b = b.scale(1/btrace)
        icfg = replace(cfg.inner, bond_cap=cfg.bond_cap, cutoff=cfg.cutoff,
                       tolerance=min(cfg.inner.tolerance, cfg.tolerance))
        rho, rep = solve_shifted(mpo, b.scale(cfg.ness_scale), 0.,
                                x0=b, config=icfg,
                                rank_one=RankOneTerm(cfg.ness_scale, b, trace))
        rho = compress(rho, cfg.bond_cap, cfg.cutoff)
        tr = inner(trace, rho)
        if abs(tr) > 1e-14:
            rho = rho.scale(1/tr)
        ness_error = eigen_residual(mpo, rho, 0.)
        trace_error = float(abs(inner(trace, rho)-1))
        history.append(dict(phase='ness', solver=asdict(rep), residual=ness_error,
                            trace_error=trace_error, max_bond=max(rho.bond_dims)))
    if n_modes == 0:
        ok = ness_error <= cfg.tolerance and trace_error <= cfg.tolerance
        return SpectrumResult(np.empty(0, complex), [], np.empty(0),
                              'converged' if ok else 'ness_not_converged', ok,
                              rho, ness_error, trace_error, history,
                              'gpu' if USE_GPU else 'cpu', time.perf_counter()-started,
                              max(rho.bond_dims))
    seed_states = _make_seeds(mpo, seeds, cfg, max(cfg.covariance_rank, n_modes+cfg.guard))
    if any(not isinstance(x, Mps) for x in seed_states):
        raise TypeError('All seeds must be native MPS')
    qn = seed_states[0].qntot
    for x in seed_states:
        if not isinstance(x, Mps) or x.pbond_list != mpo.model.pbond_list:
            raise ValueError('All seeds must be MPS with the MPO physical dimensions')
        if not np.array_equal(x.qntot, qn):
            raise ValueError('Seeds must share a QN sector; solve distinct sectors separately')
    if trace is not None and not np.array_equal(trace.qntot, qn):
        raise ValueError('Trace projection requires the trace sector; use trace=None in a traceless nonzero-charge sector')
    space = _Space(mpo, cfg, trace)
    dictionary = _Dictionary(mpo, cfg, trace)
    sources = []
    for x in seed_states:
        x = _normalized(space.projector.apply(x), cfg)
        if x is not None:
            dictionary.append(x)
            sources.append(x)
    if not sources:
        raise ValueError('All seeds vanish after trace projection')
    sources, _ = _factor_compress(sources, cfg.covariance_rank, cfg)
    extraction = dictionary.extract(n_modes+cfg.guard)
    status = 'outer_budget'
    for outer in range(cfg.max_outer):
        if cfg.max_seconds is not None and time.perf_counter()-started >= cfg.max_seconds:
            status = 'wall_time_budget'
            break
        record = dict(phase='lyapunov', outer=outer, shifted_solves=[], covariance_rank=len(sources))
        zfactors, w = [], [x.copy() for x in sources]
        last_y = [None]*len(w)
        poles = _poles(cfg, extraction['values'], outer)
        for z in poles:
            new_w = []
            for col, rhs in enumerate(w):
                y, rep = space.solve(rhs, z, x0=last_y[col])
                record['shifted_solves'].append(rep)
                if y is None:
                    new_w.append(rhs)
                    continue
                last_y[col] = y
                dictionary.append(y)
                zfactors.append(y.scale(np.sqrt(2*z.real)))
                # The ADI recurrence factors themselves are capped MPS.
                updated = combine([rhs, y], [1, -2*z.real], cfg.bond_cap, cfg.cutoff)
                new_w.append(space.projector.apply(updated))
            w = new_w
            if len(dictionary.states) >= cfg.max_subspace:
                break
        lyap_error, floor = _lyapunov_residual(space, zfactors, sources)
        record.update(lyapunov_residual=lyap_error, lyapunov_roundoff_floor=floor,
                      inverse_step_resolved=bool(lyap_error <= cfg.lyapunov_tolerance),
                      inner_rhs_solves=len(record['shifted_solves']))
        next_sources, cov = _factor_compress(zfactors, cfg.covariance_rank, cfg)
        record['covariance'] = cov
        extraction = dictionary.extract(n_modes+cfg.guard)
        record.update(values=extraction['values'].copy(), residuals=extraction['residuals'].copy(),
                      dictionary_width=len(dictionary.states), max_bond=max(max(x.bond_dims) for x in dictionary.states),
                      seconds=time.perf_counter()-started)
        history.append(record)
        if callback is not None:
            callback(record)
        have = len(extraction['values']) >= n_modes
        if have and np.all(extraction['residuals'][:n_modes] <= cfg.tolerance) and outer+1 >= cfg.min_outer:
            status = 'converged'
            break
        # Soft-lock residual-converged eigenvectors, with projection through
        # their physical Gram matrix. The projected shifted residual measures
        # errors in the approximate invariant quotient explicitly.
        converged = [x for x, e in zip(extraction['states'][:n_modes], extraction['residuals'][:n_modes]) if e <= cfg.tolerance]
        if converged and len(converged) < n_modes:
            space.locked = converged
            space.rebuild()
            record['locked_modes'] = len(converged)
            next_sources = [space.projector.apply(x) for x in next_sources]
            next_sources = [x for x in next_sources if norm(x)>1e-12]
        if not next_sources:
            next_sources = [space.projector.apply(x) for x in seed_states]
            next_sources = [x for x in next_sources if norm(x)>1e-12]
        if not next_sources:
            status = 'source_rank_exhausted'
            break
        sources, _ = _factor_compress(next_sources, cfg.covariance_rank, cfg)
        if len(dictionary.states) >= cfg.max_subspace:
            # Rebuild a real dictionary from capped target+guard and source
            # MPS. Old physical sources are released only after residual checks.
            fresh = _Dictionary(mpo, cfg, trace)
            for x in extraction['states'] + sources:
                fresh.append(x)
            trial = fresh.extract(n_modes+cfg.guard)
            if len(trial['values']) < min(n_modes, len(extraction['values'])):
                status = 'dictionary_budget'
                break
            target_count = min(n_modes+cfg.guard, len(extraction['states']))
            loss = _subspace_loss(fresh.states, extraction['states'][:target_count], cfg.gram_cutoff)
            preserved = loss <= cfg.restart_tolerance and all(trial['residuals'][i] <= extraction['residuals'][i] + cfg.restart_tolerance
                            for i in range(min(n_modes, len(trial['values']))))
            if not preserved or len(fresh.states) >= cfg.max_subspace:
                status = 'dictionary_budget'
                break
            dictionary, extraction = fresh, trial
            history.append(dict(phase='restart', dictionary_width=len(fresh.states), subspace_loss=loss))
    # Original-operator inverse refinement is explicitly distinct from the
    # Lyapunov discovery kernel. Its complex shifts may be in either half-plane.
    for iteration in range(cfg.refinement_steps):
        if status == 'converged':
            break
        if cfg.max_seconds is not None and time.perf_counter()-started >= cfg.max_seconds:
            status = 'wall_time_budget'
            break
        refinement = dict(phase='refinement', iteration=iteration, shifted_solves=[])
        fresh = _Dictionary(mpo, cfg, trace)
        for x in extraction['states']:
            fresh.append(x)
        for value, x, error in zip(extraction['values'][:n_modes], extraction['states'][:n_modes], extraction['residuals'][:n_modes]):
            if error <= cfg.tolerance:
                continue
            offset = max(cfg.tolerance*10, min(cfg.damping*0.1, max(abs(value.real)*0.1, error*0.1)))
            y, rep = solve_shifted(mpo, x, value+offset,
                                  x0=x, config=replace(cfg.inner, bond_cap=cfg.bond_cap, cutoff=cfg.cutoff))
            if trace is not None:
                y = _Projector([trace], cfg).apply(y)
            fresh.append(y)
            refinement['shifted_solves'].append(asdict(rep))
        trial = fresh.extract(n_modes+cfg.guard)
        if len(trial['values']) >= min(n_modes, len(extraction['values'])):
            extraction = trial
        refinement.update(values=extraction['values'].copy(), residuals=extraction['residuals'].copy())
        history.append(refinement)
        if len(extraction['values']) >= n_modes and np.all(extraction['residuals'][:n_modes] <= cfg.tolerance):
            status = 'converged'
    vals, states, errors = extraction['values'][:n_modes], extraction['states'][:n_modes], extraction['residuals'][:n_modes]
    # Every value corresponds to its actual returned capped MPS, with no
    # substitution of residuals measured on an untruncated implicit vector.
    errors = np.array([eigen_residual(mpo, x, v) for x, v in zip(states, vals)])
    ok = len(vals) == n_modes and np.all(errors <= cfg.tolerance)
    if len(states) > 1:
        output_gram = np.array([[inner(a, b) for b in states] for a in states])
        spectrum = la.eigvalsh((output_gram+output_gram.conj().T)/2)
        independent = spectrum[0] > cfg.gram_cutoff*spectrum[-1]
        if not independent:
            ok = False
            status = 'output_rank_deficient'
    if trace is not None:
        ok = ok and all(abs(inner(trace, x))/norm(trace) <= cfg.tolerance for x in states)
    if compute_ness:
        ness_ok = ness_error <= cfg.tolerance and trace_error <= cfg.tolerance
        ok = ok and ness_ok
        if not ness_ok:
            status = 'ness_not_converged'
    if ok:
        status = 'converged'
    elif status == 'converged':
        status = 'output_not_converged'
    backend.sync()
    return SpectrumResult(vals, states, errors, status, bool(ok), rho, ness_error,
                          trace_error, history, 'gpu' if USE_GPU else 'cpu',
                          time.perf_counter()-started,
                          max([max(x.bond_dims) for x in states] + ([max(rho.bond_dims)] if rho is not None else [0])),
                          False, 'trace-free sector' if trace is not None else 'supplied invariant sector')
