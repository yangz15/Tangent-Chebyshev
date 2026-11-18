"""Matrix-free fixed-reference TTNS Chebyshev response, one or two sites."""
import numpy as np


def chebyshev_moments(space, operator, source, bounds, actions):
    """Compute 2*actions moments with exactly `actions` projected H actions.

    `space` is a TreeTangent and `source` is its projected Vector. `bounds`
    encloses the spectrum of P H P in the operator's energy units. The fixed
    reference, exterior frames and symmetry sector do not change during
    recurrence. Returns moments, scale, source weight, and GPU stage times.
    """
    from .problems import Timer
    if not isinstance(actions, (int, np.integer)) or actions < 1:
        raise ValueError('actions must be a positive integer')
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('bounds must be a finite increasing pair')
    center, width = (low+high)/2, (high-low)/2
    weight = source.norm**2
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Projected source must have positive finite norm')
    initial = source.scale(1/np.sqrt(weight))
    previous, current = None, initial.copy()
    moments = np.empty(2*actions)
    timer = Timer()
    max_identity_error = 0.
    for n in range(actions):
        with timer.stage('projected_H_actions'):
            image = space.apply(operator, current, energy_shift=center).scale(1/width)
        with timer.stage('moment_algebra_and_gauge'):
            norm = current.dot(current).real
            if not np.isfinite(norm) or norm > 1.00001:
                raise FloatingPointError(f'Chebyshev norm guard: n={n}, norm2={norm}')
            if n and (n < 12 or n % 50 == 0):
                defect = abs(initial.dot(current).real*weight-moments[n])/weight
                max_identity_error = max(max_identity_error, defect)
                if defect > 1e-7:
                    raise FloatingPointError(f'Moment doubling identity failed: {defect}')
            following = image if n == 0 else image.scale(2).axpy(-1, previous).restore_gauge(inplace=True)
            cross = current.dot(following)
            if abs(cross.imag) > 1e-8:
                raise FloatingPointError('Complex moment from a Hermitian response operator')
            if n == 0:
                moments[:2] = [weight*norm, weight*cross.real]
            else:
                moments[2*n:2*n+2] = [2*weight*norm-moments[0], 2*weight*cross.real-moments[1]]
            previous, current = current, following
    return dict(moments=moments, center=center, width=width, source_weight=weight,
        actions=actions, computed_coefficients=len(moments), two_site=space.two_site,
        stage_seconds={k:v/1e9 for k,v in timer.ns.items()},
        max_moment_identity_error=max_identity_error, response_eigenproblem=False)


def tangent_chebyshev(reference, operator, source_operator, *, bounds, actions,
                      target=None, two_site=False, remove_reference=False):
    """Project a TTNO source and propagate in a frozen TTNS response space.

    `two_site=True` includes the complete one-site space plus the edge sector
    discarded at BOTH endpoints; it is not a two-site TDVP sweep. `target`
    specifies the source's conserved charges. `remove_reference=True` removes
    only the reference direction at the root, appropriate for inelastic IR.
    """
    from .tangent import TreeTangent
    from ._backend.gpu_charged import device_operator
    space = TreeTangent(reference, target, two_site=two_site, remove_reference=remove_reference)
    source = space.project(device_operator(source_operator), reference)
    return chebyshev_moments(space, device_operator(operator), source, bounds, actions)


def tangent_chebyshev_1site(reference, operator, source_operator, **kwargs):
    """One-site specialization of :func:`tangent_chebyshev`; same arguments."""
    return tangent_chebyshev(reference, operator, source_operator, two_site=False, **kwargs)


def tangent_chebyshev_2site(reference, operator, source_operator, **kwargs):
    """One-site plus both-end-discarded edge sectors; same arguments."""
    return tangent_chebyshev(reference, operator, source_operator, two_site=True, **kwargs)


def estimate_tangent_bounds(space, operator, *, energy_shift=0., unit=1., padding=1e-5):
    """Estimate physical projected extrema with matrix-free Lanczos.

    Gauge-complement coordinates receive a penalty away from the requested
    endpoint. Residual checks and padding give a numerical estimate, not a
    rigorous enclosure. This setup eigensolve is separate from propagation.
    """
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
    if unit <= 0 or padding <= 0:
        raise ValueError('unit and padding must be positive')
    indices = [cp.flatnonzero(m.ravel()) for m in space.masks]
    size = sum(len(i) for i in indices)
    if size < 4:
        raise ValueError('At least four masked coordinates are needed for endpoint Lanczos')
    def pack(v):
        return cp.concatenate([a.ravel()[i] for a,i in zip(v.blocks,indices)])
    def unpack(x):
        v = space.zeros()
        start = 0
        for a,i in zip(v.blocks,indices):
            a.ravel()[i] = x[start:start+len(i)]
            start += len(i)
        return v
    penalty = [0.]
    def apply(x):
        v = unpack(x).restore_gauge(inplace=True)
        return pack(space.apply(operator,v,energy_shift=energy_shift).scale(1/unit)) + penalty[0]*(x-pack(v))
    op = LinearOperator((size,size),matvec=apply,dtype=cp.complex128)
    rng = cp.random.RandomState(173)
    guess = pack(unpack(rng.normal(size=size)+1j*rng.normal(size=size)).restore_gauge(inplace=True))
    guess /= cp.linalg.norm(guess)
    def endpoint(which):
        # CuPy Lanczos may break down with NaNs when ncv exceeds the physical
        # Krylov rank, even though the redundant coordinate dimension is large.
        # Retry a smaller basis and always check finite Ritz values/residuals.
        attempts=[]
        capacities=sorted({max(3,min(n,size-1,(1024**3)//guess.nbytes))
                           for n in (64,32,16,8,4,3)},reverse=True)
        for ncv in capacities:
            try:
                # CuPy uses v0 as mutable work storage; preserve the seed for
                # retries and for the other endpoint.
                values,vectors=eigsh(op,k=1,which=which,v0=guess.copy(),ncv=ncv,maxiter=10000,tol=1e-8)
                value=float(values[0])
                residual=float(cp.linalg.norm(apply(vectors[:,0])-value*vectors[:,0]))
            except (ValueError,RuntimeError) as exc:
                attempts.append((ncv,str(exc)))
                continue
            if np.isfinite(value) and np.isfinite(residual) and residual <= max(1e-6,abs(value)*1e-7):
                return value,residual
            attempts.append((ncv,value,residual))
        raise FloatingPointError(f'No finite converged endpoint {which}; attempts={attempts}')
    # Establish a magnitude from both ends before moving the null complement.
    preliminary = [endpoint(w)[0] for w in ('SA','LA')]
    magnitude = 10*max(1.,*map(abs,preliminary))
    pairs = []
    for which in ('SA','LA'):
        penalty[0] = magnitude if which=='SA' else -magnitude
        pairs.append(endpoint(which))
    margin = max(padding*(pairs[1][0]-pairs[0][0]),100*max(r for _,r in pairs),1e-5)
    return [energy_shift+(pairs[0][0]-margin)*unit, energy_shift+(pairs[1][0]+margin)*unit]
