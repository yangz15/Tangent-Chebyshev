"""Streaming block Chebyshev moments and retarded Hamiltonian resolvents.

These routines assume a Hermitian projected Hamiltonian. In particular the
moment product identities and Hermitian whitening below do not extend to a
general complex non-Hermitian operator. Use ``ReducedTangentPencil`` in
``tangent_krylov`` with a non-Hermitian Arnoldi reduction for that case.

Only a fixed number of tangent-vector batches is live. Scalar block moments
may optionally build a canonical reduced Hamiltonian. This reduction has a
separate conditioning error and is not a substitute for tangent convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChebyshevInterval:
    lower: float
    upper: float
    edge: float = 0.98

    def __post_init__(self):
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)
                and self.lower < self.upper and 0 < self.edge < 1):
            raise ValueError("need finite ordered bounds and 0 < edge < 1")

    @property
    def center(self):
        return (self.lower+self.upper)/2

    @property
    def scale(self):
        return (self.upper-self.lower)/(2*self.edge)


def block_moments(space, mpo, source, energy, interval, steps, check_every=32, callback=None):
    """2*steps moments from steps batched H actions using product identities.

mu[n,a,b] = <source_a|T_n(H')|source_b>. Inputs need not be normalized.
Zero source columns are permitted. Checks synchronize only at checkpoints.
"""
    if (not isinstance(steps, (int, np.integer)) or steps < 1 or
            not isinstance(check_every, (int, np.integer)) or check_every < 1):
        raise ValueError("steps and check_every must be positive integers")
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    xp = space.xp
    if not source or source[0].shape[0] < 1:
        raise ValueError("need at least one source column")
    mu0 = space.gram(source, source)
    count = source[0].shape[0]
    moments = xp.empty((2*steps, count, count), dtype=xp.complex128)
    norms = xp.empty((steps+1, count), dtype=xp.float64)
    moments[0], norms[0] = mu0, xp.real(xp.diag(mu0))
    shift = energy + interval.center

    def action(t):
        return [x/interval.scale for x in space.apply(mpo, t, energy_shift=shift)]

    previous = [a.copy() for a in source]
    current = action(previous)
    mu1 = space.gram(source, current)
    moments[1] = mu1
    gram = space.gram(current, current)
    norms[1] = xp.real(xp.diag(gram))
    baseline = xp.maximum(norms[0], 1e-28)
    pair_scale = xp.sqrt(baseline[:, None]*baseline[None, :])
    checked_steps = 0
    def validate_norms(count):
        nonlocal checked_steps
        first_norm = checked_steps+1 if checked_steps else 0
        values = norms[first_norm:count+1]
        blocks = moments[2*checked_steps:2*count]
        # Reduce on device; only three scalars cross to the host per
        # checkpoint. Check every block before accepting Hermitian product
        # identities, including off-diagonal source correlations.
        status = space.host(xp.stack((
            xp.all(xp.isfinite(values)) & xp.all(xp.isfinite(blocks)),
            xp.max(values/baseline),
            xp.max(xp.abs(blocks-blocks.conj().transpose(0, 2, 1))/pair_scale))))
        if not status[0] or status[1] > 1.01:
            raise FloatingPointError("Chebyshev growth: bounds, gauge, or Hermiticity failed")
        if status[2] > 1e-8:
            raise FloatingPointError("non-Hermitian block Chebyshev moments")
        checked_steps = count
    if steps == 1 or check_every == 1:
        validate_norms(1)
        if callback is not None:
            callback(1, moments[:2], norms[:2])
    for n in range(1, steps):
        moments[2*n] = 2*gram-mu0
        following = space.project([2*x-y for x, y in zip(action(current), previous)])
        moments[2*n+1] = 2*space.gram(current, following)-mu1
        gram = space.gram(following, following)
        norms[n+1] = xp.real(xp.diag(gram))
        previous, current = current, following
        if (n+1) % check_every == 0 or n+1 == steps:
            validate_norms(n+1)
            if callback is not None:
                callback(n+1, moments[:2*(n+1)], norms[:n+2])
    return moments, norms


def _validated_moments(moments, xp):
    """Validate the Hermitian contract before forming or repairing a pencil."""
    mu = xp.asarray(moments)
    if mu.ndim != 3 or len(mu) < 1 or mu.shape[1] < 1 or mu.shape[1] != mu.shape[2]:
        raise ValueError("need nonempty square block moments")
    # A single source Gram diagonal sets the natural magnitude for every
    # Chebyshev block, even when a particular moment is zero by symmetry.
    scale = xp.maximum(xp.max(xp.abs(mu[0])), np.finfo(float).tiny)
    defect = xp.max(xp.abs(mu-mu.conj().transpose(0, 2, 1)))/scale
    if not bool(xp.all(xp.isfinite(mu))) or float(defect) > 1e-8:
        raise ValueError("block moments must be finite and Hermitian")
    return mu


def direct_resolvent(moments, omega, eta, interval, xp=np, chunk=128):
    """Retarded G(z)=B^dagger(z-H_exc)^(-1)B, plus a series-tail bound.

The bound assumes a Hermitian scaled spectrum in [-1,1]. No Jackson damping
is used: eta is the Lorentzian HWHM. Returned arrays remain in xp.
"""
    if (not np.isfinite(eta) or eta <= 0 or
            not isinstance(chunk, (int, np.integer)) or chunk < 1):
        raise ValueError("eta and chunk must be positive")
    mu = _validated_moments(moments, xp)
    omega = xp.asarray(omega)
    if mu.ndim != 3 or len(mu) < 1 or mu.shape[1] != mu.shape[2] or omega.ndim != 1:
        raise ValueError("need square block moments and a one-dimensional frequency grid")
    out = xp.empty((len(omega), *mu.shape[1:]), dtype=xp.complex128)
    bound = xp.empty(len(omega), dtype=xp.float64)
    weight = xp.real(xp.trace(mu[0]))
    for start in range(0, len(omega), chunk):
        z = (omega[start:start+chunk]+1j*eta-interval.center)/interval.scale
        root = xp.sqrt(z-1)*xp.sqrt(z+1)
        q = 1/(z+root)  # Analytic upper-half-plane branch; |q| < 1.
        # Bounded blocks turn frequency reconstruction into GPU matrix products.
        # No per-moment device synchronization or full frequency/order table.
        value = xp.zeros((len(z), mu.shape[1]*mu.shape[2]), dtype=xp.complex128)
        for first in range(1, len(mu), 2048):
            last = min(first+2048, len(mu))
            coefficients = q[:, None]**xp.arange(first, last)[None]
            value += 2*coefficients @ mu[first:last].reshape(last-first, -1)
        value = value.reshape(len(z), *mu.shape[1:])
        out[start:start+chunk] = (value+mu[0])/(interval.scale*root[:, None, None])
        radius = xp.abs(q)
        bound[start:start+chunk] = (2*weight*radius**len(mu)
            /(interval.scale*xp.abs(root)*(1-radius)))
    return out, bound


def canonical_resolvent(moments, omega, eta, interval, dimension,
                        xp=np, cutoff=1e-10, chunk=128):
    """Block Chebyshev moment pencil, canonical whitening and reduced resolvent."""
    if (not np.isfinite(eta) or eta <= 0 or not 0 < cutoff < 1 or
            not isinstance(dimension, (int, np.integer)) or dimension < 1 or
            not isinstance(chunk, (int, np.integer)) or chunk < 1):
        raise ValueError("invalid eta, cutoff, or dimension")
    mu = _validated_moments(moments, xp)
    omega = xp.asarray(omega)
    if omega.ndim != 1:
        raise ValueError("need a one-dimensional frequency grid")
    if len(mu) < 2*dimension:
        raise ValueError("need at least 2*dimension block moments")
    n = xp.arange(dimension)
    def overlap(j):
        return (mu[n[:, None]+j[None, :]]+mu[xp.abs(n[:, None]-j[None, :])])/2
    s4 = overlap(n)
    h4 = (overlap(n+1)+overlap(xp.abs(n-1)))/2
    ns = mu.shape[1]
    s = s4.transpose(0, 2, 1, 3).reshape(dimension*ns, dimension*ns)
    h = h4.transpose(0, 2, 1, 3).reshape(dimension*ns, dimension*ns)
    s, h = (s+s.conj().T)/2, (h+h.conj().T)/2
    se, su = xp.linalg.eigh(s)
    largest = float(se[-1])
    if largest <= 0 or not np.isfinite(largest):
        raise ValueError("all sources are zero or nonfinite")
    if float(se[0]) < -1e-9*largest:
        raise FloatingPointError("indefinite Chebyshev Gram matrix")
    keep = se > cutoff*largest
    x = su[:, keep]/xp.sqrt(se[keep])[None]
    heff = x.conj().T @ h @ x
    defect = float(xp.linalg.norm(x.conj().T @ s @ x-xp.eye(x.shape[1])))
    if not np.isfinite(defect) or defect > 1e-4:
        raise FloatingPointError("canonical metric is ill-conditioned; increase cutoff")
    heff = (heff+heff.conj().T)/2
    poles, u = xp.linalg.eigh(heff)
    if float(xp.max(xp.abs(poles))) > 1.000001:
        raise FloatingPointError("reduced poles leave the scaling interval")
    amplitudes = u.conj().T @ x.conj().T @ mu[:dimension].reshape(dimension*ns, ns)
    energies = interval.center+interval.scale*poles
    out = xp.empty((len(omega), ns, ns), dtype=xp.complex128)
    for start in range(0, len(omega), chunk):
        inverse = 1/(omega[start:start+chunk, None]+1j*eta-energies[None])
        out[start:start+chunk] = xp.einsum("ra,fr,rb->fab", amplitudes.conj(), inverse,
                                         amplitudes, optimize=True)
    report = dict(rank=int(x.shape[1]), pencil_dimension=dimension*ns,
        canonical_cutoff=cutoff, metric_defect=defect,
        condition=float(se[-1]/se[keep][0]),
        reduced_source_retention=float(xp.sum(xp.abs(amplitudes)**2)/xp.trace(mu[0]).real))
    return out, energies, amplitudes, report
