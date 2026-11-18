"""Burg spectral maximum entropy from Chebyshev/cosine covariances.

This is the covariance extension problem solved by Levinson--Durbin. It is
not Burg's forward/backward residual estimator for sampled time series.
"""
import numpy as np


def burg_maximum_entropy(moments, loading=1e-10):
    """Return prediction coefficients and positivity diagnostics.

    Parameters
    ----------
    moments : one-dimensional real array
        Undamped cosine moments mu[0], ..., mu[M-1], with mu[0] > 0.
    loading : float
        Nonnegative relative diagonal loading of the covariance Toeplitz
        matrix. Only its zeroth covariance is changed during the solve.

    Returns
    -------
    prediction, diagnostics
        Predicted mu[n] = sum(prediction[j-1] * mu[n-j], j=1,...,M-1).
        Reflection coefficients must remain inside the unit disk. No pole
        clipping, root repair, or Hamiltonian diagonalization is performed.
    """
    raw = np.asarray(moments)
    if raw.ndim != 1 or len(raw) < 2 or np.iscomplexobj(raw):
        raise ValueError('At least two real moments are required')
    if not np.isfinite(raw).all() or raw[0] <= 0 or loading < 0 or not np.isfinite(loading):
        raise ValueError('Require finite moments, positive mu0 and nonnegative loading')
    r = raw.astype(np.longdouble)
    r[0] *= 1 + np.longdouble(loading)
    a = np.ones(1, dtype=np.longdouble)
    variance = r[0]
    reflections = []
    for k in range(1, len(r)):
        reflection = -(r[k] + np.dot(a[1:], r[k-1:0:-1])) / variance
        if not np.isfinite(reflection) or abs(reflection) >= 1:
            raise ValueError(f'Covariance positivity failed at order {k}: {reflection}')
        previous = a
        a = np.empty(k+1, dtype=np.longdouble)
        a[0], a[-1] = 1, reflection
        a[1:-1] = previous[1:] + reflection * previous[:0:-1]
        variance *= 1 - reflection**2
        reflections.append(float(reflection))
    return -a[1:], dict(order=len(a)-1, loading=loading,
        prediction_variance=float(variance), largest_reflection=max(map(abs, reflections)),
        coefficient_l1=float(np.sum(abs(a))))


def extend_moments(moments, prediction, length):
    """Continue scalar moments to `length`; predictions are not new H actions."""
    mu = np.asarray(moments, dtype=np.longdouble)
    coefficient = np.asarray(prediction, dtype=np.longdouble)
    if mu.ndim != 1 or coefficient.ndim != 1 or not 1 <= len(coefficient) < len(mu):
        raise ValueError('Prediction order must be between one and len(moments)-1')
    if not isinstance(length, (int, np.integer)) or length < len(mu):
        raise ValueError('length must be an integer at least len(moments)')
    output = np.empty(length, dtype=np.longdouble)
    output[:len(mu)] = mu
    for n in range(len(mu), length):
        output[n] = np.dot(coefficient, output[n-len(coefficient):n][::-1])
        if not np.isfinite(output[n]) or abs(output[n]) > abs(output[0])*1.0001:
            raise ValueError(f'Predicted moment bound violated at {n}: {output[n]}')
    return np.asarray(output, dtype=np.float64)


def reconstruct_spectrum(moments, center, width, grid, *, kernel, broadening,
                         loading=1e-10, continuation=32000):
    """Reconstruct a physically broadened Burg spectrum on `grid`.

    Energies, center, width and broadening must share units. Gaussian uses
    scalar continuation followed by DCT quadrature. Lorentzian evaluates
    the rational generating function, without an eigenproblem. Returns a
    NumPy curve and diagnostics. Requires a CUDA device.
    """
    import cupy as cp
    from .kernels import gaussian
    from .rational import lorentz_rational
    mu = np.asarray(moments, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if width <= 0 or broadening <= 0 or grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError('Positive scale/broadening and an increasing grid are required')
    prediction, info = burg_maximum_entropy(mu/mu[0], loading)
    if kernel == 'gaussian':
        if continuation < 2*len(mu):
            raise ValueError('Gaussian continuation must be at least twice the input length')
        predicted = extend_moments(mu/mu[0], prediction, continuation)*mu[0]
        curve, count = gaussian(predicted, center, width, cp.asarray(grid), broadening)
        shorter, _ = gaussian(predicted[:continuation//2], center, width, cp.asarray(grid), broadening)
        error = float(cp.linalg.norm(curve-shorter)/cp.linalg.norm(curve))
        if error > 1e-4:
            raise FloatingPointError(f'Gaussian continuation requires more moments: {error}')
        info.update(continuation=continuation, extension_relative_error=error, quadrature_nodes=count)
    elif kernel == 'lorentzian':
        curve, _ = lorentz_rational(mu, prediction, center, width, cp.asarray(grid), broadening)
    else:
        raise ValueError('kernel must be gaussian or lorentzian')
    if not bool(cp.isfinite(curve).all()) or float(cp.min(curve)/cp.max(curve)) < -1e-5:
        raise FloatingPointError('Invalid or negative reconstructed spectrum')
    return cp.asnumpy(curve), info
