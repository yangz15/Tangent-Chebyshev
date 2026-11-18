"""TTNS tangent and compressed Chebyshev spectral projection.

The Hamiltonian action is the tree generalization of arXiv:2510.07279,
SI S-1 B (S8--S10), supplied by :class:`TTNSTangentSpace`. Canonical
orthogonalization follows Jiang et al., JPCL 12, 9344 (2021), Eqs. 7--9.
The tangent recurrence has no compression or iterative orthogonalization.
The compressed recurrence accepts a backend-resident variational fit callback.
Gaussian ``sigma`` is a standard deviation.
"""
from dataclasses import dataclass
import numpy as np

from renormalizer.mps.matrix import asnumpy


@dataclass
class ReducedSpectrum:
    poles: object
    weights: object
    coefficients: object
    overlap_eigenvalues: object
    diagnostics: dict
    canonical_transform: object = None
    effective_hamiltonian: object = None
    effective_source: object = None


@dataclass
class TangentChebyshevResult:
    overlap: object
    hamiltonian: object
    source_overlaps: object
    moments: object
    spectrum: ReducedSpectrum
    center: float
    scale: float
    source_norm2: float
    vectors: object = None


def _scalar(value):
    if np.isscalar(value):
        return np.asarray(value).item()
    return np.asarray(asnumpy(value)).item()


def _hermitian(matrix, name, module, tolerance):
    norm = float(_scalar(module.linalg.norm(matrix)))
    defect = float(_scalar(module.linalg.norm(matrix-matrix.conj().T)))
    relative = defect / max(norm, np.finfo(float).tiny)
    if not np.isfinite(relative) or relative > tolerance:
        raise ValueError(f"{name} is not Hermitian: relative defect {relative:.3e}")
    return (matrix+matrix.conj().T)*.5, relative


def _check_metric(values, count):
    largest, smallest = float(_scalar(values[-1])), float(_scalar(values[0]))
    if largest <= 0 or smallest < -max(1e-10, 100*count*np.finfo(float).eps)*largest:
        raise ValueError(f"overlap is zero or indefinite: min={smallest:.6g}, max={largest:.6g}")
    return largest, smallest


def solve_reduced_spectrum(overlap, hamiltonian, source_overlaps, *,
                           module=np, rcond=1e-10, hermitian_tolerance=1e-8):
    """Solve H c = S c e on S's positive support, weights = |c^dagger b|².

    H is in physical energy units and b_i=<t_i|source>. Negative metric
    eigenvalues beyond roundoff and non-Hermitian input are errors, rather
    than being silently repaired. All large algebra stays on ``module``.
    The reported residual is in the retained canonical subspace; discarded
    overlap directions are not part of this generalized eigenproblem.
    """
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie in (0, 1)")
    s, h, b = (module.asarray(a) for a in (overlap, hamiltonian, source_overlaps))
    if s.ndim != 2 or s.shape[0] != s.shape[1] or h.shape != s.shape or b.shape != (len(s),):
        raise ValueError("incompatible reduced matrix/source dimensions")
    if not len(s) or not bool(_scalar(module.all(module.isfinite(b)))):
        raise ValueError("empty or nonfinite reduced problem")
    s, sdef = _hermitian(s, "overlap", module, hermitian_tolerance)
    h, hdef = _hermitian(h, "Hamiltonian", module, hermitian_tolerance)
    values, u = module.linalg.eigh(s)
    largest, smallest = _check_metric(values, len(s))
    keep = values > rcond*largest
    x = u[:, keep] / module.sqrt(values[keep])[None, :]
    projected = x.conj().T @ h @ x
    projected = (projected+projected.conj().T)*.5
    effective_source = x.conj().T @ b
    poles, rotation = module.linalg.eigh(projected)
    coefficients = x @ rotation
    weights = module.abs(coefficients.conj().T @ b)**2
    metric_error = float(_scalar(module.linalg.norm(x.conj().T@s@x-module.eye(x.shape[1]))))
    residual = x.conj().T@(h@coefficients-(s@coefficients)*poles[None, :])
    residual_norm = float(_scalar(module.linalg.norm(residual)))
    if not np.isfinite(metric_error) or metric_error > 1e-4:
        raise FloatingPointError("canonical metric is ill-conditioned; increase rcond")
    return ReducedSpectrum(poles, weights, coefficients, values, dict(
        retained_rank=x.shape[1], primitive_vectors=len(s), rcond=rcond,
        overlap_min=smallest, overlap_max=largest,
        overlap_hermiticity=sdef, hamiltonian_hermiticity=hdef,
        metric_identity_error=metric_error, retained_residual_norm=residual_norm,
        weight_sum=float(_scalar(weights.sum()))), x, projected, effective_source)


def reduced_chebyshev_moments(spectrum, count, bounds, *, module=np, margin=.95):
    """Jiang's SECOND expansion, after canonical orthogonalization (Eq. 8).

    Generate mu_n = y^dagger T_n((H_eff-center)/scale) y, where
    X=U_keep/sqrt(s_keep), H_eff=X^dagger H X and y=X^dagger b. This small,
    uncompressed linear recurrence restores the Chebyshev identities inside
    the retained Krylov space. ``count`` is its moment count L, independent
    of the N primitive, possibly compressed TTNS vectors. It adds no physical
    TTNS basis vectors or missing Hilbert-space information.

    ``bounds`` are in the same physical energy convention as the reduced
    poles (including any E0 subtraction). They must enclose all reduced
    poles. No Jackson damping or broadening is applied to the returned
    moments; direct pole broadening remains the preferred exact reduced-
    model route for a fixed Gaussian sigma. Return (moments, center, scale).
    """
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi, margin]).all() or lo >= hi or not 0 < margin < 1:
        raise ValueError('finite ordered bounds and margin in (0, 1) required')
    if spectrum.effective_hamiltonian is None or spectrum.effective_source is None:
        raise ValueError('canonical reduced Hamiltonian and source are required')
    h, source = (module.asarray(a) for a in
                 (spectrum.effective_hamiltonian, spectrum.effective_source))
    poles = module.asarray(spectrum.poles)
    tolerance = 1e-12*max(1., abs(lo), abs(hi))
    if float(_scalar(poles.min())) < lo-tolerance or float(_scalar(poles.max())) > hi+tolerance:
        raise ValueError('bounds do not enclose the reduced poles')
    center, scale = (hi+lo)/2, (hi-lo)/(2*margin)
    previous, current = None, source.copy()
    mu = module.empty(count, dtype=module.complex128)
    for n in range(count):
        mu[n] = module.vdot(source, current)
        if n+1 < count:
            action = (h@current-center*current)/scale
            previous, current = current, action if n == 0 else 2*action-previous
    checks = module.stack((module.max(module.abs(mu.imag)), module.max(module.abs(mu.real))))
    weight = float(_scalar(module.vdot(source, source).real))
    if (not bool(_scalar(module.all(module.isfinite(checks)))) or
            float(_scalar(checks[0])) > 1e-8*max(weight, np.finfo(float).tiny) or
            float(_scalar(checks[1])) > weight*(1+1e-7)):
        raise FloatingPointError('unstable or non-real reduced Chebyshev recurrence')
    return mu.real.copy(), center, scale


def chebyshev_moment_matrices(moments, count, *, module=np):
    """S and scaled H from 2N moments of an EXACT fixed linear recurrence.

    With individually compressed vectors this is an approximation, not their
    Gram/Ritz matrix, and may be indefinite. ``compressed_chebyshev`` exposes
    this experimental route with the same strict metric checks as tangent
    Chebyshev. No negative metric directions are silently repaired.
    """
    mu = module.asarray(moments)
    if count < 1 or mu.ndim != 1 or len(mu) < 2*count:
        raise ValueError("2*count moments are required")
    i, j = module.arange(count)[:, None], module.arange(count)[None, :]
    s = .5*(mu[i+j]+mu[module.abs(i-j)])
    h = .25*(mu[i+j+1]+mu[module.abs(i+j-1)]
             +mu[module.abs(i-j+1)]+mu[module.abs(i-j-1)])
    return s, h


def _record_moment_pair(mu, n, norm2, cross):
    """Shared doubled-moment convention; cross = <t_n|t_{n+1}>."""
    if n == 0:
        mu[0], mu[1] = norm2.real, cross.real
    else:
        mu[2*n] = 2*norm2.real-mu[0]
        mu[2*n+1] = 2*cross.real-mu[1]


def compressed_chebyshev(source, orders, bounds, *, step, inner_products,
                         hamiltonian_elements=None, module=np, energy_shift=0.,
                         margin=.95, rcond=1e-10, matrix_mode="explicit",
                         keep_vectors=False, callback=None):
    """Yield reduced spectra at requested compressed-TTNS recurrence orders.

    ``step(n, current, previous, center, scale)`` returns a NEW fitted state
    t_{n+1} approximating A*t_0 (n=0) or 2*A*t_n-t_{n-1}, where
    A=(H-center)/scale. It must not mutate either input or normalize the
    result. The full sum must be fitted together. ``inner_products(bras, ket)``
    returns <bra|ket> on ``module``; ``hamiltonian_elements(bras, ket)`` returns
    <bra|H|ket> for the absolute, unscaled Hermitian H. Callbacks permit
    resident GPU fits without an enlarged H|ket> tensor network.

    ``explicit`` (default) batches actual pair contractions, retaining N
    states and requiring N-1 recurrence fits. ``moments`` uses exactly the
    tangent doubled-moment/matrix/solver route: N fits, N primitive vectors
    plus the scratch t_N, and only three live recurrence states. Compression
    makes this an additional approximation. An indefinite moment metric is
    an ERROR, with no clipping or automatic expensive fallback. Positive
    metric support alone does not establish accuracy of this approximation.
    Power-of-two prefix checks starting at N=8 reject invalid metrics early.

    ``orders`` is an integer or an iterable of positive integers. Results are
    yielded in increasing order. ``callback(n, current)`` observes each
    primitive vector (not scratch t_N) and must not mutate it. All scalar
    arrays and reduced algebra stay on ``module``. No reorthogonalization or
    normalization is introduced here.
    """
    if isinstance(orders, (int, np.integer)):
        orders = [orders]
    orders = list(orders)
    if not orders or any(not isinstance(n, (int, np.integer)) or n < 1 for n in orders):
        raise ValueError("orders must contain positive integers")
    orders = sorted(set(orders))
    if matrix_mode not in ("moments", "explicit"):
        raise ValueError("matrix_mode must be moments or explicit")
    if matrix_mode == "explicit" and hamiltonian_elements is None:
        raise ValueError("explicit mode requires hamiltonian_elements")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi, energy_shift, margin]).all() or lo >= hi or not 0 < margin < 1:
        raise ValueError("finite ordered bounds and margin in (0, 1) required")
    scale, center = (hi-lo)/(2*margin), (hi+lo)/2
    weight = float(_scalar(inner_products([source], source)[0].real))
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("source has zero or nonfinite norm")
    count = orders[-1]
    explicit = matrix_mode == "explicit"
    vectors = [] if explicit or keep_vectors else None
    mu = None if explicit else module.empty(2*count, dtype=module.float64)
    s = module.empty((count, count), dtype=module.complex128) if explicit else None
    h = module.empty_like(s) if explicit else None
    previous, current = None, source
    for n in range(count):
        if vectors is not None:
            vectors.append(current)
        if callback is not None:
            callback(n, current)
        if explicit:
            scol = module.asarray(inner_products(vectors, current))
            hcol = module.asarray(hamiltonian_elements(vectors, current))
            if scol.shape != (n+1,) or hcol.shape != (n+1,):
                raise ValueError("pair callbacks must return one element per bra")
            checks = module.stack((module.abs(scol[-1].imag), module.abs(hcol[-1].imag)))
            if (not bool(_scalar(module.all(module.isfinite(scol)))) or
                    not bool(_scalar(module.all(module.isfinite(hcol)))) or
                    float(_scalar(checks[0])) > 1e-8*weight or
                    float(_scalar(checks[1])) > 1e-8*max(weight, abs(_scalar(hcol[-1])))):
                raise FloatingPointError("nonfinite or non-real Hermitian pair element")
            s[:n+1, n], h[:n+1, n] = scol, hcol
            s[n, :n+1], h[n, :n+1] = scol.conj(), hcol.conj()
        else:
            following = step(n, current, previous, center, scale)
            norm2, cross = inner_products([current], current)[0], inner_products([current], following)[0]
            _record_moment_pair(mu, n, norm2, cross)
            checks = module.stack((norm2.real, module.abs(norm2.imag),
                                  cross.real, module.abs(cross.imag)))
            if (not bool(_scalar(module.all(module.isfinite(checks)))) or
                    float(_scalar(module.max(checks[[1, 3]]))) > 1e-8*weight):
                raise FloatingPointError("non-real Hermitian Chebyshev moment after compression")
        size = n+1
        if not explicit and size >= 8 and size & (size-1) == 0 and size not in orders:
            metric, _ = chebyshev_moment_matrices(mu, size, module=module)
            try:
                _check_metric(module.linalg.eigvalsh(metric), size)
            except ValueError as exc:
                raise ValueError(
                    f"compressed moment overlap is indefinite at N={size}; "
                    f"compression broke the moment identities ({exc}). "
                    "Use matrix_mode='explicit' or improve the recurrence fits; "
                    "no spectrum was repaired.") from exc
        if size in orders:
            if explicit:
                sh = s[:size, :size].copy()
                hh = h[:size, :size].copy()-energy_shift*sh
                b = sh[:, 0].copy()
            else:
                sh, scaled_h = chebyshev_moment_matrices(mu, size, module=module)
                hh = scale*scaled_h+(center-energy_shift)*sh
                b = mu[:size].copy()
            try:
                spectrum = solve_reduced_spectrum(sh, hh, b, module=module, rcond=rcond)
            except ValueError as exc:
                if not explicit and "indefinite" in str(exc):
                    raise ValueError(
                        f"compressed moment overlap is indefinite at N={size}; "
                        f"compression broke the moment identities ({exc}). Use matrix_mode='explicit' "
                        "or improve the recurrence fits; no spectrum was repaired.") from exc
                raise
            if spectrum.diagnostics['weight_sum'] > weight*(1+1e-5):
                raise FloatingPointError("reduced spectral weights violate the source sum rule")
            spectrum.diagnostics.update(source_norm2=weight,
                lost_source_weight=weight-spectrum.diagnostics['weight_sum'],
                matrix_mode=matrix_mode, moment_approximation=not explicit,
                recurrence_fits=size-1 if explicit else size,
                pair_operator_elements=size*(size+1)//2 if explicit else 0)
            yield TangentChebyshevResult(sh, hh, b, None if explicit else mu[:2*size].copy(),
                spectrum, center, scale, weight, list(vectors) if keep_vectors else None)
        if explicit and size < count:
            following = step(n, current, previous, center, scale)
        if not explicit or size < count:
            previous, current = current, following


def tangent_chebyshev(space, operator, source, count, bounds, *, energy_shift=0.,
                      margin=.95, rcond=1e-10, matrix_mode="moments",
                      keep_vectors=False, callback=None, check_every=32):
    """Generate T_n((P H P-center)/scale)|source> and its Ritz spectrum.

    ``bounds`` enclose absolute eigenenergies of P H P; ``energy_shift``
    (usually E0) is subtracted ONLY when constructing the physical reduced
    Hamiltonian. The source is gauge-projected once without renormalization.
    ``moments`` retains three working vectors and needs N H applications.
    ``explicit`` contracts the stored vectors and their H images directly,
    providing an independent audit of the moment identities. Returned arrays
    live on the tangent space backend. An optional callback receives (n, t_n)
    and must not mutate the vectors.
    Norm and Hermiticity checks synchronize every ``check_every`` actions and
    at completion, retaining diagnostics for every intermediate vector.
    """
    if (not isinstance(count, (int, np.integer)) or count < 1 or
            not isinstance(check_every, (int, np.integer)) or check_every < 1):
        raise ValueError("count and check_every must be positive integers")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi, energy_shift, margin]).all() or lo >= hi or not 0 < margin < 1:
        raise ValueError("finite ordered bounds and margin in (0, 1) required")
    if matrix_mode not in ("moments", "explicit"):
        raise ValueError("matrix_mode must be moments or explicit")
    a, center = (hi-lo)/(2*margin), (hi+lo)/2
    module = space.array_module
    source = source.restore_gauge()
    weight = source.dot(source).real
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("source has zero or nonfinite norm")
    def action(v):
        return space.apply(operator, v, energy_shift=center).scale(1/a, inplace=True)
    vectors = [] if keep_vectors or matrix_mode == "explicit" else None
    images = [] if matrix_mode == "explicit" else None
    mu = module.empty(2*count, dtype=module.float64)
    mu[0] = weight
    checks = module.empty((min(count, check_every), 3), dtype=module.float64)
    current_norm = module.asarray(weight)
    previous, current = None, source
    for n in range(count):
        if vectors is not None:
            vectors.append(current)
        if callback is not None:
            callback(n, current)
        image = action(current)
        if images is not None:
            images.append(image)
        following = image if n == 0 else image.scale(2).axpy(-1., previous).restore_gauge(inplace=True)
        cross = space._dot_value(current, following)
        _record_moment_pair(mu, n, current_norm, cross)
        # |T_n(x)|<=1 on [-1,1]. A growing norm reveals unsafe scaling,
        # non-Hermiticity, or numerical loss in the fixed tangent metric.
        current_norm = space._dot_value(following, following)
        slot = n % check_every
        checks[slot] = module.stack((current_norm.real, module.abs(current_norm.imag),
                                     module.abs(cross.imag)))
        if slot+1 == check_every or n+1 == count:
            checked = asnumpy(checks[:slot+1])
            if not np.isfinite(checked).all() or np.any(checked[:, 1:] > 1e-8*weight):
                raise FloatingPointError("non-real or nonfinite Hermitian Chebyshev moment")
            unstable = np.flatnonzero(checked[:, 0] > weight*(1+1e-5))
            if len(unstable):
                order = n-slot+int(unstable[0])+1
                raise FloatingPointError(f"unstable Chebyshev vector {order}; widen spectral bounds")
        previous, current = current, following
    if matrix_mode == "moments":
        s, h = chebyshev_moment_matrices(mu, count, module=module)
        b = mu[:count].copy()
    else:
        # One pair of GEMMs per node instead of O(count**2) scalar
        # contractions/kernel launches. The parent metric is essential for
        # charged tangent sectors and non-root-canonical coordinates.
        s = module.zeros((count, count), dtype=module.complex128)
        h = module.zeros_like(s)
        for node, metric in enumerate(space.parent_metrics):
            parent_dim = vectors[0].tensors[node].shape[-1]
            left = module.stack([v.tensors[node] for v in vectors]).reshape(count, -1, parent_dim)
            right = module.stack([v.tensors[node] for v in images]).reshape(count, -1, parent_dim)
            bra = left.reshape(count, -1).conj()
            s += bra @ (left @ metric.T).reshape(count, -1).T
            h += bra @ (right @ metric.T).reshape(count, -1).T
        b = s[:, 0].copy()
    physical_h = a*h+(center-energy_shift)*s
    spectrum = solve_reduced_spectrum(s, physical_h, b, module=module, rcond=rcond)
    if spectrum.diagnostics['weight_sum'] > weight*(1+1e-5):
        raise FloatingPointError("reduced spectral weights violate the source sum rule")
    spectrum.diagnostics.update(source_norm2=weight,
        lost_source_weight=weight-spectrum.diagnostics["weight_sum"],
        matrix_mode=matrix_mode, h_actions=count)
    return TangentChebyshevResult(s, physical_h, b, mu, spectrum, center, a,
                                 weight, vectors if keep_vectors else None)


def gaussian_spectrum(grid, poles, weights, sigma, *, module=np, chunk_size=1024):
    """Unit-area Gaussians; sigma is the standard deviation in grid units."""
    if not np.isfinite(sigma) or sigma <= 0 or chunk_size < 1:
        raise ValueError("positive finite sigma and chunk_size required")
    grid, poles, weights = (module.asarray(a) for a in (grid, poles, weights))
    if grid.ndim != 1 or poles.ndim != 1 or poles.shape != weights.shape:
        raise ValueError("grid, poles and weights must be compatible vectors")
    result = module.zeros(grid.shape, dtype=module.float64)
    for start in range(0, len(poles), chunk_size):
        z = (grid[:, None]-poles[None, start:start+chunk_size])/sigma
        result += module.exp(-.5*z*z) @ weights[start:start+chunk_size]
    return result/(np.sqrt(2*np.pi)*sigma)


def gaussian_time_spectrum(grid, times, correlation, sigma, *, module=np, chunk_size=256):
    """Re/pi integral_0^T exp(i*w*t-sigma²*t²/2) C(t) dt.

    This is precisely the same Gaussian as ``gaussian_spectrum`` in the
    infinite-time limit. No extra normalization of the correlation is made.
    """
    times_host = np.asarray(asnumpy(times), dtype=float)
    if (len(times_host) < 2 or times_host[0] != 0 or np.any(np.diff(times_host) <= 0)
            or not np.isfinite(times_host).all() or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("times must increase from zero; sigma must be positive")
    grid, times, correlation = (module.asarray(a) for a in (grid, times, correlation))
    if grid.ndim != 1 or correlation.shape != times.shape:
        raise ValueError("incompatible grid/correlation dimensions")
    dt = module.diff(times)
    quadrature = module.concatenate((dt[:1]/2, (dt[:-1]+dt[1:])/2, dt[-1:]/2))
    weighted = correlation*module.exp(-.5*(sigma*times)**2)*quadrature/np.pi
    return module.concatenate([(module.exp(1j*grid[k:k+chunk_size, None]*times)@weighted).real
                               for k in range(0, len(grid), chunk_size)])
