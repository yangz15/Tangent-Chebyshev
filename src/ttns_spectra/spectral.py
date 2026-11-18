"""Canonical coCheb reduction and physical pole broadening."""
import numpy as np
import cupy as cp
def host(a):
    return cp.asnumpy(a)

def canonical_model(overlap, projected, weight, cutoff=1e-12):
    """Canonical metric treatment of b†(zS-H)^(-1)b on retained support."""
    s = (overlap+overlap.conj().T)*.5
    h = (projected+projected.conj().T)*.5
    if not bool(cp.all(cp.isfinite(s))) or not bool(cp.all(cp.isfinite(h))):
        raise FloatingPointError('Nonfinite projected matrices')
    h_defect = float(cp.linalg.norm(projected-projected.conj().T)
                     /max(float(cp.linalg.norm(projected)), 1e-300))
    if h_defect > 1e-8:
        raise FloatingPointError(f'Non-Hermitian projected Hamiltonian: {h_defect}')
    se, su = cp.linalg.eigh(s)
    threshold = max(1e-14, cutoff*float(se[-1]))
    if float(se[0]) < -1e-10*float(se[-1]):
        raise FloatingPointError('Indefinite overlap matrix')
    keep = se > threshold
    if not bool(cp.any(keep)):
        raise FloatingPointError('No retained metric support')
    x = su[:, keep]/cp.sqrt(se[keep])[None, :]
    heff = x.conj().T @ h @ x
    source = x.conj().T @ s[:, 0]
    defect = float(cp.linalg.norm(x.conj().T @ s @ x-cp.eye(x.shape[1])))
    if defect > .005:
        raise FloatingPointError(f'Canonical metric identity defect {defect}')
    anti = float(cp.linalg.norm(heff-heff.conj().T))
    heff = (heff+heff.conj().T)*.5
    poles, rotation = cp.linalg.eigh(heff)
    weights = cp.abs(rotation.conj().T @ source)**2*weight
    if float(cp.max(cp.abs(poles))) >= 1.0:
        raise FloatingPointError('Reduced spectrum leaves Chebyshev scaling interval')
    report = dict(basis_size=s.shape[0], retained_rank=x.shape[1], overlap_relative_cutoff=cutoff,
        overlap_absolute_threshold=threshold, overlap_min=float(se[0]), overlap_max=float(se[-1]),
        condition_number=float(se[-1]/se[keep][0]), metric_identity_defect=defect,
        projected_antihermitian_relative=h_defect,
        reduced_antihermitian_absolute=anti, source_capture=float(cp.sum(weights)/weight),
        stepwise_reorthogonalization=False, per_vector_normalization=False,
        canonical_overlap_filtering=True)
    return poles, weights, heff, source, report


def broaden_poles(poles_cm, weights, grid, sigma_cm, lorentz=False):
    grid = cp.asarray(grid)
    result = cp.empty(len(grid))
    for start in range(0, len(grid), 128):
        delta = grid[start:start+128, None]-poles_cm[None, :]
        kernel = (sigma_cm/(cp.pi*(delta**2+sigma_cm**2)) if lorentz else
                  cp.exp(-.5*(delta/sigma_cm)**2)/(np.sqrt(2*np.pi)*sigma_cm))
        result[start:start+128] = kernel @ weights
    return host(result)
