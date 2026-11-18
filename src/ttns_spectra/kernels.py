"""Direct physical broadening from raw Chebyshev moments, without S/H.

Lorentz: analytic Chebyshev resolvent expansion with retarded branch.
Gaussian: integrate the finite series by Chebyshev-Gauss quadrature (DCT).
No additional Jackson/Lorentz damping and no clipping of negative tails.
All arguments use one common energy unit within a call.
"""
import numpy as np
import cupy as cp
from cupyx.scipy.fft import dct


def lorentz(mu, center, scale, grid, eta):
    mu, grid = cp.asarray(mu), cp.asarray(grid)
    z = (grid-center+1j*eta)/scale
    root = cp.sqrt(z*z-1)
    root = cp.where(cp.abs(z-root) < 1, root, -root)
    q = 1/(z+root)
    assert float(cp.max(cp.abs(q))) < 1
    # Horner evaluation of mu0 + 2*sum(mu_n*q**n).
    value = cp.zeros_like(q)
    for n in range(len(mu)-1, 0, -1):
        value = (value+2*mu[n])*q
    green = (value+mu[0])/(scale*root)
    # Conditional analytic bound if the operator spectrum is within [-1,1].
    tail = 2*abs(mu[0])*cp.abs(q)**len(mu)/(scale*cp.abs(root)*(1-cp.abs(q)))/np.pi
    return -green.imag/np.pi, green, tail


def gaussian(mu, center, scale, grid, sigma, refinement=1):
    mu, grid = cp.asarray(mu), cp.asarray(grid)
    need = max(4*len(mu), 6*np.pi*scale/sigma, 8192)
    count = refinement*(1 << int(np.ceil(np.log2(need))))
    coefficients = cp.zeros(count, dtype=cp.float64)
    coefficients[:len(mu)] = mu.real
    weights = dct(coefficients, type=3)/count
    nodes = center+scale*cp.cos(np.pi*(cp.arange(count)+.5)/count)
    keep = (nodes >= grid[0]-10*sigma) & (nodes <= grid[-1]+10*sigma)
    nodes, weights = nodes[keep], weights[keep]
    result = cp.empty(len(grid))
    for start in range(0, len(grid), 128):
        delta = (grid[start:start+128, None]-nodes)/sigma
        result[start:start+128] = cp.exp(-.5*delta**2)@weights/(np.sqrt(2*np.pi)*sigma)
    return result, count


def pole_moments_for_diagnostic(poles_scaled, weights, count):
    """Replay only already-saved poles; not input to any production spectrum."""
    nodes, weights = cp.asarray(poles_scaled), cp.asarray(weights)
    assert float(cp.max(cp.abs(nodes))) < 1
    return cp.cos(cp.arange(count)[:, None]*cp.arccos(nodes))@weights
