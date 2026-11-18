"""Memory-bounded GPU Lanczos exponentials and controlled partial SVD.

The Krylov basis is stored only while it fits the byte budget. Large local
vectors use a second Lanczos pass to reconstruct the result from scalar
coefficients, rather than storing m full vectors. All physical arrays stay
on GPU. Small tridiagonal diagonalizations use SciPy.
"""
import numpy as np
import cupy as cp
from scipy.linalg import eigh_tridiagonal

_subtract = cp.ElementwiseKernel(
    'T q, T previous, float64 alpha, float64 beta', 'T image',
    'image -= alpha * q + beta * previous;', 'xy_lanczos_subtract')
_accumulate = cp.ElementwiseKernel(
    'T q, complex128 coefficient', 'complex128 result',
    'result += coefficient * q;', 'xy_lanczos_accumulate')


def coefficients(alpha, beta, step):
    poles, rotation = eigh_tridiagonal(np.asarray(alpha), np.asarray(beta))
    return rotation @ (np.exp(step * poles) * rotation[0])


def expm_bounded(action, step, start, rtol=1e-7, basis_bytes=256*1024**2,
                 max_order=80):
    start = cp.asarray(start).ravel()
    norm = float(cp.linalg.norm(start))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError('Invalid exponential source norm')
    capacity = max(0, int(basis_bytes // start.nbytes))
    stored = []
    q = start.copy()
    q /= norm
    previous = None
    alpha, beta = [], []
    old_coeff = None
    actions = 0
    shift = None
    converged = False
    for j in range(min(max_order, start.size)):
        if capacity and j < capacity:
            stored.append(q)
        else:
            # Switch to two-pass reconstruction without a block reallocation.
            stored.clear()
            capacity = 0
        image = action(q)
        actions += 1
        rayleigh = float(cp.vdot(q, image).real)
        if shift is None:
            shift = rayleigh
        alpha.append(rayleigh-shift)
        _subtract(q, q if previous is None else previous, rayleigh,
                  0. if previous is None else beta[-1], image)
        next_beta = float(cp.linalg.norm(image))
        coeff = coefficients(alpha, beta, step)
        if old_coeff is not None:
            delta = np.linalg.norm(coeff[:-1]-old_coeff)**2 + abs(coeff[-1])**2
            converged = bool(np.sqrt(delta) <= rtol * np.linalg.norm(coeff))
        breakdown = next_beta <= 1e-13 * max(1., abs(rayleigh))
        if (j >= 3 and converged) or breakdown or j+1 == start.size:
            break
        beta.append(next_beta)
        previous, q = q, image
        q /= next_beta
        old_coeff = coeff
    else:
        raise RuntimeError(f'Local exponential did not converge in {max_order} Lanczos vectors')
    order = len(alpha)
    coeff *= norm * np.exp(step*shift)
    del q, previous, image
    result = cp.zeros(start.shape, dtype=cp.complex128)
    if capacity:
        for weight, vector in zip(coeff, stored):
            _accumulate(vector, complex(weight), result)
        second_pass = False
    else:
        q = start.copy()
        q /= norm
        previous = None
        for j in range(order):
            _accumulate(q, complex(coeff[j]), result)
            if j+1 == order:
                break
            image = action(q)
            actions += 1
            _subtract(q, q if previous is None else previous, alpha[j]+shift,
                      0. if previous is None else beta[j-1], image)
            previous, q = q, image
            q /= beta[j]
        second_pass = True
    return result, dict(vectors=order, h_actions=actions, two_pass=second_pass,
                        basis_bytes=0 if second_pass else order*start.nbytes,
                        rayleigh_shift=shift)


def partial_svd(matrix, rank, tolerance=1e-8, oversample=32, max_iterations=6):
    """Adaptive randomized range SVD, with an explicit singular residual.

    Never form a full square U/V when only rank columns are needed. A fixed
    seed makes the probe reproducible. Power steps are orthogonalized to
    resolve the retained small Schmidt values. The residual is checked
    against the matrix norm; failure is reported instead of accepted.
    """
    rows, columns = matrix.shape
    width = min(min(rows, columns), rank+oversample)
    rng = cp.random.RandomState(721)
    omega = rng.standard_normal((columns, width)).astype(matrix.dtype)
    if cp.iscomplexobj(matrix):
        omega += 1j*rng.standard_normal((columns, width))
    q, _ = cp.linalg.qr(matrix @ omega, mode='reduced')
    norm = float(cp.linalg.norm(matrix))
    for iteration in range(max_iterations+1):
        projected = q.conj().T @ matrix
        # Extract right trial directions from a small Gram matrix, then QR
        # and SVD a width-by-rank core. The final factors are orthonormal;
        # no division by small Gram eigenvalues is used. This avoids the
        # expensive SVD of a wide width-by-thousands matrix.
        _, rotation = cp.linalg.eigh(projected @ projected.conj().T)
        right, _ = cp.linalg.qr(projected.conj().T @ rotation[:, -rank:], mode='reduced')
        small_u, singular, small_vh = cp.linalg.svd(projected @ right, full_matrices=False)
        u = q @ small_u
        vh = small_vh @ right.conj().T
        residual = float(cp.linalg.norm(matrix @ vh.conj().T-u*singular[None, :]))
        relative = residual/max(norm, 1e-30)
        if relative <= tolerance:
            return u, singular, vh, dict(solver='partial_range_svd',
                subspace_iterations=iteration, relative_singular_residual=relative,
                oversampling=width-rank)
        if iteration != max_iterations:
            z, _ = cp.linalg.qr(matrix.conj().T @ q, mode='reduced')
            q, _ = cp.linalg.qr(matrix @ z, mode='reduced')
    if oversample < 128 and width < min(rows, columns):
        return partial_svd(matrix, rank, tolerance=tolerance,
                           oversample=min(128, 2*oversample), max_iterations=10)
    raise RuntimeError(f'Partial SVD singular residual {relative:.3g} exceeds {tolerance:.3g}')
