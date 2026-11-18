"""Exact contractions and PEPS-restricted tangent Krylov on two-row strips.

Column blocking is a contraction representation only. Ground-state variables
are the original rank-five PEPS tensors. The column-chain tangent space is
restricted to the image of their Jacobian before every Hamiltonian action.
No full Hilbert-space wavefunction is constructed. Costs still grow rapidly
with strip width (only width two is supported) and with PEPS bond dimension.
"""

import numpy as np
import opt_einsum as oe
from scipy.optimize import minimize

from renormalizer.mps import Mps
from renormalizer.mps.tangent import (
    MpsTangentSpace, _left_message, _right_message, _effective_one_site,
)
from .peps import Peps, positive_int
from .tangent import lanczos_spectrum


def _contract(subscripts, *arrays):
    return oe.contract(subscripts, *arrays, optimize="greedy")


class StripPepsLinearization:
    """Column contractions and derivatives with respect to PEPS site entries.

    Physical column order is (top, bottom); columns run from left to right.
    Parameters remain in PEPS row-major order. Bond weights are absorbed and
    each site is balanced by a scalar before defining these coordinates.
    """

    def __init__(self, state):
        if state.shape[0] != 2:
            raise ValueError("exact strip contraction currently requires two rows")
        self.shape, self.sites = state.shape, state.sites
        self.ncolumns = state.shape[1]
        arrays = state.absorbed_tensors()
        self.tensor_shapes = [arrays[s].shape for s in self.sites]
        balanced = []
        for site in self.sites:
            norm = np.linalg.norm(arrays[site])
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("zero or nonfinite site tensor")
            balanced.append(arrays[site] / norm)
        self.parameters = np.concatenate([a.ravel() for a in balanced])
        offsets = np.cumsum([0] + [a.size for a in balanced])
        self.slices = [slice(offsets[i], offsets[i + 1]) for i in range(len(balanced))]
        self.nparams = len(self.parameters)

    def arrays(self, parameters=None):
        parameters = self.parameters if parameters is None else np.asarray(parameters)
        if parameters.shape != (self.nparams,) or not np.all(np.isfinite(parameters)):
            raise ValueError("parameters must be a finite vector of the correct length")
        return [parameters[s].reshape(shape) for s, shape in zip(self.slices, self.tensor_shapes)]

    def columns(self, parameters=None):
        arrays = self.arrays(parameters)
        columns = []
        for c in range(self.ncolumns):
            top, bottom = arrays[c][:, 0], arrays[c + self.ncolumns][:, :, :, 0]
            block = _contract("prvl,qvsb->lbpqrs", top, bottom)
            l, b, p, q, r, s = block.shape
            columns.append(block.reshape(l * b, p * q, r * s))
        return columns

    def column_derivative(self, column, parameters=None):
        """Derivative with a global parameter index on the last axis."""
        arrays = self.arrays(parameters)
        c = column
        top, bottom = arrays[c][:, 0], arrays[c + self.ncolumns][:, :, :, 0]
        dtop = np.eye(top.size).reshape(top.shape + (top.size,))
        dbottom = np.eye(bottom.size).reshape(bottom.shape + (bottom.size,))
        a = _contract("prvln,qvsb->lbpqrsn", dtop, bottom)
        b = _contract("prvl,qvsbn->lbpqrsn", top, dbottom)
        l, lb, p, q, r, rb = a.shape[:-1]
        shape = (l * lb, p * q, r * rb)
        out = np.zeros(shape + (self.nparams,), dtype=np.result_type(top, bottom))
        out[..., self.slices[c]] = a.reshape(shape + (top.size,))
        out[..., self.slices[c + self.ncolumns]] = b.reshape(shape + (bottom.size,))
        return out

    def pullback(self, cotangents, parameters=None):
        arrays = self.arrays(parameters)
        out = np.zeros(self.nparams, dtype=np.result_type(*arrays, *cotangents))
        for c, gradient in enumerate(cotangents):
            top, bottom = arrays[c][:, 0], arrays[c + self.ncolumns][:, :, :, 0]
            p, r, v, l = top.shape
            q, _, s, b = bottom.shape
            gradient = gradient.reshape(l, b, p, q, r, s)
            out[self.slices[c]] = _contract("lbpqrs,qvsb->prvl", gradient, bottom.conj()).ravel()
            out[self.slices[c + self.ncolumns]] = _contract("lbpqrs,prvl->qvsb", gradient, top.conj()).ravel()
        return out

    def to_peps(self, parameters=None):
        arrays = self.arrays(parameters)
        return Peps([arrays[:self.ncolumns], arrays[self.ncolumns:]])

    def to_column_mps(self, model, parameters=None):
        """Exact contraction container; never an independent variational ansatz."""
        columns = self.columns(parameters)
        if len(model.basis) != self.ncolumns or any(np.any(b.sigmaqn) for b in model.basis):
            raise ValueError("column model must have one uncharged basis per column")
        out = Mps()
        out.model = model
        out.dtype = np.result_type(*columns)
        for tensor in columns:
            out.append(tensor)
        out.build_empty_qn()
        return out


def strip_energy_gradient(linear, hamiltonian, parameters=None):
    """Exact Rayleigh quotient and real/complex PEPS parameter gradient.

    Returns (energy, gradient, squared_norm). For complex parameters the real
    differential is Re(vdot(gradient, dparameters)). Requires a Hermitian MPO.
    """
    columns = linear.columns(parameters)
    operators = [np.asarray(t.array) for t in hamiltonian]
    if len(operators) != len(columns):
        raise ValueError("Hamiltonian must act on the blocked columns")
    identity = [np.eye(c.shape[1]).reshape(1, c.shape[1], c.shape[1], 1) for c in columns]
    lefts, rights = [], []
    for ops in (operators, identity):
        left = [np.ones((1, 1, 1))]
        for c, w in zip(columns, ops):
            left.append(_left_message(left[-1], c, w, c))
        right = [None] * (len(columns) + 1)
        right[-1] = np.ones((1, 1, 1))
        for k in range(len(columns) - 1, -1, -1):
            right[k] = _right_message(right[k + 1], columns[k], ops[k], columns[k])
        lefts.append(left)
        rights.append(right)
    norm = float(lefts[1][-1].item().real)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("zero or nonfinite PEPS norm")
    energy = float(lefts[0][-1].item().real / norm)
    gradients = []
    for k, c in enumerate(columns):
        acted = _effective_one_site(lefts[0][k], operators[k], rights[0][k + 1], c)
        normal = _effective_one_site(lefts[1][k], identity[k], rights[1][k + 1], c)
        gradients.append(2 * (acted - energy * normal) / norm)
    return energy, linear.pullback(gradients, parameters), norm


def optimize_peps_strip(state, hamiltonian, maxiter=500, gtol=1e-8, ftol=1e-14,
                        callback=None):
    """Optimize original PEPS tensors using exact strip energies/gradients."""
    linear = StripPepsLinearization(state)
    iscomplex = np.iscomplexobj(linear.parameters)
    x0 = (np.concatenate([linear.parameters.real, linear.parameters.imag])
          if iscomplex else linear.parameters.copy())

    def decode(x):
        return x[:linear.nparams] + 1j * x[linear.nparams:] if iscomplex else x

    def objective(x):
        energy, gradient, _ = strip_energy_gradient(linear, hamiltonian, decode(x))
        return energy, (np.concatenate([gradient.real, gradient.imag]) if iscomplex else gradient.real)

    def report(x):
        if callback is not None:
            callback(linear, decode(x))

    result = minimize(objective, x0, method="L-BFGS-B", jac=True, callback=report,
                      options=dict(maxiter=positive_int(maxiter, "maxiter"), gtol=gtol,
                                   ftol=ftol, maxls=40, maxcor=30))
    parameters = decode(result.x)
    _, _, norm = strip_energy_gradient(linear, hamiltonian, parameters)
    out = linear.to_peps(parameters)
    out.tensors[(0, 0)] /= np.sqrt(norm)
    return out, result


class StripPepsTangent:
    """Orthogonal PEPS tangent projector with exact two-row contractions.

    Canonical column-chain coordinates provide a Euclidean contraction frame.
    SVD of the PEPS Jacobian in that frame removes gauge/null directions.
    Its retained left singular vectors restrict both sides of every action.
    This is NOT the full tangent space of an unconstrained blocked MPS.
    The dense coordinate Jacobian limits length/bond dimension, but its size
    is unrelated to the exponentially large physical Hilbert dimension.
    """

    def __init__(self, state, hamiltonian, metric_cutoff=1e-10,
                 max_jacobian_entries=2 ** 27):
        if not np.isfinite(metric_cutoff) or not 0 < metric_cutoff < 1:
            raise ValueError("metric_cutoff must be in (0, 1)")
        self.linearization = linear = StripPepsLinearization(state)
        self.hamiltonian = hamiltonian
        self.energy, _, norm = strip_energy_gradient(linear, hamiltonian)
        self.frame = MpsTangentSpace(linear.to_column_mps(hamiltonian.model))
        self.complements, sizes = [], []
        for a, shape in zip(self.frame.left, self.frame.shapes):
            matrix = a.reshape(-1, a.shape[-1])
            q = np.linalg.qr(matrix, mode="complete")[0][:, matrix.shape[1]:]
            self.complements.append(q)
            sizes.append(q.shape[1] * shape[-1])
        offsets = np.cumsum([0] + sizes)
        self.coordinate_slices = [slice(offsets[k], offsets[k + 1]) for k in range(len(sizes))]
        self.ambient_dimension = int(offsets[-1])
        if self.ambient_dimension * linear.nparams > positive_int(max_jacobian_entries, "max_jacobian_entries"):
            raise ValueError("strip coordinate Jacobian exceeds max_jacobian_entries")
        columns = linear.columns()
        dtype = np.result_type(*columns)
        n, p = len(columns), linear.nparams
        left, dleft = [np.ones((1, 1))], [np.zeros((1, 1, p), dtype=dtype)]
        for k in range(n):
            a, c, dc = self.frame.left[k].conj(), columns[k], linear.column_derivative(k)
            dleft.append(_contract("abn,apc,bpd->cdn", dleft[-1], a, c)
                         + _contract("ab,apc,bpdn->cdn", left[-1], a, dc))
            left.append(_contract("ab,apc,bpd->cd", left[-1], a, c))
        right, dright = [None] * (n + 1), [None] * (n + 1)
        right[-1], dright[-1] = np.ones((1, 1)), np.zeros((1, 1, p), dtype=dtype)
        for k in range(n - 1, -1, -1):
            b, c, dc = self.frame.right[k].conj(), columns[k], linear.column_derivative(k)
            dright[k] = (_contract("apc,bpd,cdn->abn", b, c, dright[k + 1])
                         + _contract("apc,bpdn,cd->abn", b, dc, right[k + 1]))
            right[k] = _contract("apc,bpd,cd->ab", b, c, right[k + 1])
        blocks = []
        for k in range(n):
            c, dc = columns[k], linear.column_derivative(k)
            g = (_contract("abn,bpd,cd->apcn", dleft[k], c, right[k + 1])
                 + _contract("ab,bpdn,cd->apcn", left[k], dc, right[k + 1])
                 + _contract("ab,bpd,cdn->apcn", left[k], c, dright[k + 1])) / np.sqrt(norm)
            q = self.complements[k]
            block = _contract("ax,acn->xcn", q.conj(), g.reshape(q.shape[0], g.shape[2], p))
            blocks.append(block.reshape(sizes[k], p))
        self.tangent_map = np.concatenate(blocks, axis=0)
        u, self.singular_values, vh = np.linalg.svd(self.tangent_map, full_matrices=False)
        keep = self.singular_values > np.sqrt(metric_cutoff) * self.singular_values[0]
        if not np.any(keep):
            raise ValueError("empty PEPS tangent space")
        self.basis = u[:, keep]
        self.whitener = vh[keep].conj().T / self.singular_values[keep]
        self.rank = self.basis.shape[1]
        self.metric_cutoff = float(metric_cutoff)
        self.metric_condition = float((self.singular_values[0] / self.singular_values[keep][-1]) ** 2)

    def pack(self, vector):
        return np.concatenate([(q.conj().T @ t.reshape(q.shape[0], t.shape[-1])).ravel()
                               for q, t in zip(self.complements, vector.tensors)])

    def unpack(self, coordinates):
        coordinates = np.asarray(coordinates)
        if coordinates.shape != (self.ambient_dimension,):
            raise ValueError("column coordinate shape mismatch")
        tensors = [(q @ coordinates[s].reshape(q.shape[1], shape[-1])).reshape(shape)
                   for q, s, shape in zip(self.complements, self.coordinate_slices, self.frame.shapes)]
        return self.frame.vector(tensors, project=False)

    def matvec(self, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (self.rank,):
            raise ValueError("PEPS tangent coefficient shape mismatch")
        vector = self.unpack(self.basis @ coefficients)
        acted = self.frame.apply(self.hamiltonian, vector, energy_shift=self.energy)
        return self.basis.conj().T @ self.pack(acted)

    def source(self, probe):
        """PEPS coordinates of the connected probe applied to the reference."""
        return self.basis.conj().T @ self.pack(self.frame.project_reference(probe))

    def spectrum(self, probe, max_steps=None, tol=1e-12):
        return lanczos_spectrum(self.matvec, self.source(probe), max_steps, tol)

    def shutdown(self):
        self.frame.shutdown()
