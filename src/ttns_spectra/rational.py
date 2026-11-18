"""Evaluate the Green function of a moment recurrence without roots/eigenvalues."""
import numpy as np
import cupy as cp


def coefficients(mu, prediction):
    # mu_n = sum_j prediction[j-1] * mu_(n-j), n>=len(mu).
    # F(q)=sum_n mu_n q^n = numerator(q)/denominator(q).
    mu = np.asarray(mu, dtype=np.longdouble)
    denominator = np.concatenate((np.ones(1, dtype=np.longdouble), -np.asarray(prediction, dtype=np.longdouble)))
    assert len(denominator) == len(mu)
    numerator = np.convolve(mu, denominator)[:len(mu)]
    return numerator, denominator


def lorentz_rational(mu, prediction, center, scale, grid, eta):
    numerator, denominator = coefficients(mu, prediction)
    z = (cp.asarray(grid)-center+1j*eta)/scale
    root = cp.sqrt(z*z-1)
    root = cp.where(cp.abs(z-root) < 1, root, -root)
    q = 1/(z+root)
    assert float(cp.max(cp.abs(q))) < 1
    nvalue, dvalue = cp.zeros_like(q), cp.zeros_like(q)
    # These are scalar polynomial coefficients, not a reduced Hamiltonian.
    for n, d in zip(numerator[::-1], denominator[::-1]):
        nvalue = nvalue*q+float(n)
        dvalue = dvalue*q+float(d)
    green = (2*nvalue/dvalue-float(mu[0]))/(scale*root)
    return -green.imag/np.pi, green
