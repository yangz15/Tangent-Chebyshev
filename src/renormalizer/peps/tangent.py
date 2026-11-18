"""Exact finite-system PEPS tangent reference and projected spectral Lanczos.

This module deliberately constructs full-Hilbert-space vectors and Jacobians.
It is a correctness/physics prototype, not a scalable CTMRG implementation.
Lanczos avoids constructing the projected Hamiltonian, but not the dense
tangent basis. Simple-update weights are absorbed before differentiating.
"""

import numpy as np
import opt_einsum as oe
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize

from .peps import Peps, bond_axes, positive_int


class PepsLinearization:
    """Analytic derivatives of the unnormalized PEPS wavefunction.

    All entries of ordinary, independently varying site tensors are parameters.
    Explicit bond weights are first absorbed into the site tensors, and are not
    treated as additional parameters. The complex derivative is holomorphic.
    The size guards limit outputs; contraction intermediates can be larger.
    """

    def __init__(self, state, max_hilbert=4096, max_jacobian_entries=2 ** 25):
        self.shape, self.sites = state.shape, state.sites
        self.tensor_shapes = [state.tensors[s].shape for s in self.sites]
        self.hilbert_dim = 1
        for shape in self.tensor_shapes:
            self.hilbert_dim *= shape[0]
        self.nparams = sum(int(np.prod(shape)) for shape in self.tensor_shapes)
        if self.hilbert_dim > positive_int(max_hilbert, "max_hilbert"):
            raise ValueError("dense tangent Hilbert space exceeds max_hilbert")
        if self.hilbert_dim * self.nparams > positive_int(max_jacobian_entries, "max_jacobian_entries"):
            raise ValueError("dense Jacobian exceeds max_jacobian_entries")
        arrays = state.absorbed_tensors()
        # Scalar tensor balancing leaves the normalized physical state unchanged
        # and improves conditioning of the parameter metric.
        balanced = []
        for site in self.sites:
            norm = np.linalg.norm(arrays[site])
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("zero or nonfinite site tensor")
            balanced.append(arrays[site] / norm)
        self.parameters = np.concatenate([a.reshape(-1) for a in balanced])
        offsets = np.cumsum([0] + [a.size for a in balanced])
        self.slices = [slice(offsets[i], offsets[i + 1]) for i in range(len(balanced))]
        labels = {s: [i] + [None] * 4 for i, s in enumerate(self.sites)}
        next_label = len(self.sites)
        for a, b in state.bonds:
            ia, ib = bond_axes(a, b)
            labels[a][ia] = labels[b][ib] = next_label
            next_label += 1
        for site in self.sites:
            for axis in range(1, 5):
                if labels[site][axis] is None:
                    labels[site][axis] = next_label
                    next_label += 1
        self._labels = [labels[s] for s in self.sites]
        output = list(range(len(self.sites)))
        self._state_expression = self._expression(self._labels, self.tensor_shapes, output)
        self._jacobian_expressions = []
        for i, site_labels in enumerate(self._labels):
            others = [j for j in range(len(self.sites)) if j != i]
            inputs = [self._labels[j] for j in others]
            shapes = [self.tensor_shapes[j] for j in others]
            p = self.tensor_shapes[i][0]
            constants = [np.eye(p)]
            inputs.append([site_labels[0], next_label])
            shapes.append((p, p))
            present = set(label for labels_j in inputs for label in labels_j)
            for label, dimension in zip(site_labels[1:], self.tensor_shapes[i][1:]):
                if label not in present:
                    inputs.append([label])
                    shapes.append((dimension,))
                    constants.append(np.ones(dimension))
            expression = self._expression(inputs, shapes, output + [next_label] + site_labels[1:])
            self._jacobian_expressions.append((others, constants, expression))

    @staticmethod
    def _expression(inputs, shapes, output):
        encode = lambda labels: "".join(oe.get_symbol(label) for label in labels)
        subscripts = ",".join(encode(labels) for labels in inputs) + "->" + encode(output)
        return oe.contract_expression(subscripts, *shapes, optimize="greedy")

    def arrays(self, parameters=None):
        parameters = self.parameters if parameters is None else np.asarray(parameters)
        if parameters.shape != (self.nparams,) or not np.all(np.isfinite(parameters)):
            raise ValueError("parameters must be a finite vector of the correct length")
        return [parameters[s].reshape(shape) for s, shape in zip(self.slices, self.tensor_shapes)]

    def vector(self, parameters=None):
        return self._state_expression(*self.arrays(parameters)).reshape(-1)

    def jacobian(self, parameters=None):
        arrays = self.arrays(parameters)
        blocks = []
        for others, constants, expression in self._jacobian_expressions:
            block = expression(*([arrays[j] for j in others] + constants))
            blocks.append(block.reshape(self.hilbert_dim, -1))
        return np.concatenate(blocks, axis=1)

    def to_peps(self, parameters=None):
        arrays = self.arrays(parameters)
        nc = self.shape[1]
        return Peps([arrays[r * nc:(r + 1) * nc] for r in range(self.shape[0])])


def _checked_hamiltonian(hamiltonian, state, size):
    if hasattr(hamiltonian, "to_sparse"):
        hamiltonian.check_state(state)
        hamiltonian = hamiltonian.to_sparse(max_size=size)
    if sparse.issparse(hamiltonian):
        hamiltonian = hamiltonian.tocsr().copy()
        values = hamiltonian.data
        defect = hamiltonian - hamiltonian.conj().T
        error = np.max(abs(defect.data)) if defect.nnz else 0.0
    else:
        hamiltonian = np.array(hamiltonian, copy=True)
        if hamiltonian.shape != (size, size):
            raise ValueError("Hamiltonian shape does not match Hilbert space")
        values = hamiltonian
        error = np.max(abs(hamiltonian - hamiltonian.conj().T))
    if hamiltonian.shape != (size, size) or not np.all(np.isfinite(values)):
        raise ValueError("Hamiltonian must be finite with matching shape")
    if error > 1e-12 * max(1., np.max(abs(values)) if values.size else 0.):
        raise ValueError("Hamiltonian must be Hermitian")
    return hamiltonian


def optimize_peps_dense(state, hamiltonian, maxiter=500, gtol=1e-9, ftol=1e-14,
                        max_hilbert=4096, max_jacobian_entries=2 ** 25):
    """Small-system variational refinement using exact energy and gradient.

    Returns ``(new_state, scipy.optimize.OptimizeResult)``. Real initial tensors
    stay real; complex initial tensors optimize both real and imaginary parts.
    A stopped optimizer is not necessarily converged: inspect success, jac,
    and the independently evaluated projected ground-state residual.
    """
    linear = PepsLinearization(state, max_hilbert, max_jacobian_entries)
    matrix = _checked_hamiltonian(hamiltonian, state, linear.hilbert_dim)
    complex_parameters = np.iscomplexobj(linear.parameters)
    x0 = (np.concatenate((linear.parameters.real, linear.parameters.imag))
          if complex_parameters else linear.parameters.copy())

    def decode(x):
        return x[:linear.nparams] + 1j * x[linear.nparams:] if complex_parameters else x

    def objective(x):
        parameters = decode(x)
        vector = linear.vector(parameters)
        norm = np.vdot(vector, vector).real
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise ValueError("zero or nonfinite state during optimization")
        acted = matrix @ vector
        energy = float(np.vdot(vector, acted).real / norm)
        gradient = 2 * linear.jacobian(parameters).conj().T @ (acted - energy * vector) / norm
        gradient = (np.concatenate((gradient.real, gradient.imag))
                    if complex_parameters else gradient.real)
        return energy, gradient

    result = minimize(objective, x0, method="L-BFGS-B", jac=True,
                      options={"maxiter": positive_int(maxiter, "maxiter"),
                               "gtol": gtol, "ftol": ftol, "maxls": 40, "maxcor": 30})
    out = linear.to_peps(decode(result.x))
    norm = np.linalg.norm(linear.vector(decode(result.x)))
    out.tensors[(0, 0)] /= norm
    return out, result


class DiscreteSpectrum:
    """Nonnegative discrete spectral weights and Lorentzian broadening."""

    def __init__(self, energies, weights):
        self.energies = np.asarray(energies, dtype=float).copy()
        self.weights = np.asarray(weights, dtype=float).copy()
        if (self.energies.ndim != 1 or self.weights.shape != self.energies.shape or
                not np.all(np.isfinite(self.energies)) or not np.all(np.isfinite(self.weights)) or
                np.any(self.weights < 0)):
            raise ValueError("spectral energies/weights must be finite vectors with nonnegative weights")

    def broaden(self, frequencies, eta):
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
            raise ValueError("frequencies must be a finite vector")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        return np.sum(self.weights[:, None] * eta / np.pi /
                      ((frequencies[None, :] - self.energies[:, None]) ** 2 + eta ** 2), axis=0)

    def moments(self, order=2):
        # Moments refer to the unbroadened measure; Lorentzian tails do not
        # possess ordinary higher moments over the entire real axis.
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        return np.array([np.dot(self.weights, self.energies ** n) for n in range(order + 1)])


def lanczos_spectrum(matvec, source, max_steps=None, tol=1e-12):
    """Hermitian Lanczos with two-pass full reorthogonalization.

    The returned DiscreteSpectrum also contains alpha, beta, krylov_basis,
    orthogonality_error, and tail_norm. No state compression is performed.
    ``matvec`` must represent a fixed linear Hermitian operator.
    """
    source = np.asarray(source)
    if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError("source must be a finite nonempty vector")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    weight = float(np.vdot(source, source).real)
    if weight == 0:
        raise ValueError("source has zero spectral weight")
    steps = source.size if max_steps is None else min(source.size, positive_int(max_steps, "max_steps"))
    basis = np.zeros((source.size, steps), dtype=np.result_type(source, complex))
    basis[:, 0] = source / np.sqrt(weight)
    alpha, beta = [], []
    tail_norm = 0.0
    for i in range(steps):
        q = basis[:, i]
        acted = np.asarray(matvec(q))
        if acted.shape != source.shape or not np.all(np.isfinite(acted)):
            raise ValueError("invalid matvec output")
        diagonal = np.vdot(q, acted)
        if abs(diagonal.imag) > 1e-10 * max(1., np.linalg.norm(acted)):
            raise ValueError("matvec is not Hermitian to numerical precision")
        alpha.append(float(diagonal.real))
        residual = acted - alpha[-1] * q
        if i:
            residual -= beta[-1] * basis[:, i - 1]
        active = basis[:, :i + 1]
        for _ in range(2):
            residual -= active @ (active.conj().T @ residual)
        tail_norm = float(np.linalg.norm(residual))
        if tail_norm <= tol * max(1., np.linalg.norm(acted)) or i == steps - 1:
            break
        beta.append(tail_norm)
        basis[:, i + 1] = residual / tail_norm
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    if alpha.size == 1:
        energies, vectors = alpha.copy(), np.ones((1, 1))
    else:
        energies, vectors = eigh_tridiagonal(alpha, beta)
    out = DiscreteSpectrum(energies, weight * abs(vectors[0]) ** 2)
    out.alpha, out.beta = alpha, beta
    out.krylov_basis = basis[:, :len(alpha)]
    out.orthogonality_error = float(np.linalg.norm(
        out.krylov_basis.conj().T @ out.krylov_basis - np.eye(len(alpha))))
    out.tail_norm = tail_norm
    return out


class PepsTangent:
    """Exact one-site tangent space orthogonal to a finite PEPS reference.

    SVD of T=(I-|psi><psi|)J/||psi_raw|| removes gauge and ground-state
    directions without squaring the condition number. ``metric_cutoff`` is
    relative to the largest eigenvalue of S=T^dagger T, not its singular value.
    U_r is the orthonormal physical basis; W=V_r/s_r is the whitening map.
    Dense arrays make this class suitable only for small-system validation.
    """

    def __init__(self, state, hamiltonian, metric_cutoff=1e-10,
                 max_hilbert=4096, max_jacobian_entries=2 ** 25):
        self.linearization = PepsLinearization(state, max_hilbert, max_jacobian_entries)
        vector = self.linearization.vector()
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("zero or nonfinite PEPS norm")
        self.psi = vector / norm
        self.hamiltonian = _checked_hamiltonian(hamiltonian, state, len(vector))
        self.energy = float(np.vdot(self.psi, self.hamiltonian @ self.psi).real)
        jacobian = self.linearization.jacobian() / norm
        self.tangent_map = jacobian - np.outer(self.psi, self.psi.conj() @ jacobian)
        self._u, self.singular_values, self._vh = np.linalg.svd(self.tangent_map, full_matrices=False)
        self.set_metric_cutoff(metric_cutoff)

    def set_metric_cutoff(self, metric_cutoff):
        if not np.isfinite(metric_cutoff) or not 0 < metric_cutoff < 1:
            raise ValueError("metric_cutoff must be in (0, 1)")
        self.metric_cutoff = float(metric_cutoff)
        keep = self.singular_values > np.sqrt(metric_cutoff) * self.singular_values[0]
        if not np.any(keep):
            raise ValueError("empty tangent space")
        self.basis = self._u[:, keep]
        self.whitener = self._vh[keep].conj().T / self.singular_values[keep]
        self.rank = self.basis.shape[1]
        self.metric_condition = float((self.singular_values[0] / self.singular_values[keep][-1]) ** 2)

    def connected_source(self, source):
        """Remove the PEPS reference component from a full-Hilbert-space vector."""
        source = np.asarray(source)
        if source.shape != self.psi.shape or not np.all(np.isfinite(source)):
            raise ValueError("source must match the full Hilbert space")
        return source - self.psi * np.vdot(self.psi, source)

    def matvec(self, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (self.rank,):
            raise ValueError("tangent coefficient shape mismatch")
        vector = self.basis @ coefficients
        return self.basis.conj().T @ (self.hamiltonian @ vector) - self.energy * coefficients

    def projected_hamiltonian(self):
        matrix = self.basis.conj().T @ (self.hamiltonian @ self.basis) - self.energy * np.eye(self.rank)
        return (matrix + matrix.conj().T) / 2

    def diagonalize(self, source):
        """Explicit projected diagonalization for comparison with Lanczos."""
        coordinates = self.basis.conj().T @ self.connected_source(source)
        energies, vectors = np.linalg.eigh(self.projected_hamiltonian())
        return DiscreteSpectrum(energies, abs(vectors.conj().T @ coordinates) ** 2)

    def spectrum(self, source, max_steps=None, tol=1e-12):
        """Projected Lanczos; does not construct the dense projected Hamiltonian."""
        coordinates = self.basis.conj().T @ self.connected_source(source)
        return lanczos_spectrum(self.matvec, coordinates, max_steps, tol)

    def diagnostics(self, source):
        """Source loss, ground residual, moments, and the exact leakage identity.

        The leakage identity is evaluated for the retained projected source,
        so it remains valid even when the metric cutoff removes source weight.
        Ground-state error and tangent projection error are separate quantities.
        """
        source = self.connected_source(source)
        coordinates = self.basis.conj().T @ source
        projected_source = self.basis @ coordinates
        acted = self.hamiltonian @ source - self.energy * source
        projected_acted = self.hamiltonian @ projected_source - self.energy * projected_source
        tangent_acted = self.matvec(coordinates)
        leakage = projected_acted - self.basis @ tangent_acted
        residual = self.hamiltonian @ self.psi - self.energy * self.psi
        norm2 = lambda x: float(np.vdot(x, x).real)
        full_moments = [norm2(source), float(np.vdot(source, acted).real), norm2(acted)]
        tangent_moments = [norm2(coordinates), float(np.vdot(coordinates, tangent_acted).real), norm2(tangent_acted)]
        deficit = norm2(projected_acted) - norm2(tangent_acted)
        return {"rank": self.rank, "hilbert_dim": len(self.psi),
                "raw_parameters": self.linearization.nparams,
                "metric_cutoff": self.metric_cutoff, "metric_condition": self.metric_condition,
                "ground_energy": self.energy, "ground_residual_norm": float(np.linalg.norm(residual)),
                "ground_variance": norm2(residual),
                "tangent_ground_residual_norm": float(np.linalg.norm(self.basis.conj().T @ residual)),
                "source_norm_squared": norm2(source),
                "source_projection_loss_squared": norm2(source - projected_source),
                "full_source_moments": full_moments, "tangent_moments": tangent_moments,
                "leakage_squared": norm2(leakage),
                "second_moment_deficit_projected_source": deficit,
                "leakage_identity_error": abs(deficit - norm2(leakage)),
                "basis_orthogonality_error": float(np.linalg.norm(self.basis.conj().T @ self.basis - np.eye(self.rank))),
                "ground_orthogonality_error": float(np.linalg.norm(self.psi.conj() @ self.basis))}
