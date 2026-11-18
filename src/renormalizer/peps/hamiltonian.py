"""Local 2D Hamiltonians and a bridge from Renormalizer Model/Op objects."""

import numpy as np
from scipy import sparse

from .peps import lattice_shape, sites, bonds, bond_axes, positive_int


class LocalHam2D:
    """Sum of Hermitian one-site and nearest-neighbor two-site matrices.

    ``h1`` maps coordinates to matrices; ``h2`` maps coordinate pairs to
    matrices in that pair's order. ``phys_dim`` is an integer or a site dict.
    Terms on reversed bonds are transposed into canonical row-major order.
    """

    def __init__(self, shape, h1=None, h2=None, phys_dim=2):
        self.shape = lattice_shape(shape)
        self.sites = sites(self.shape)
        self.physical_dims = {s: positive_int(phys_dim[s] if isinstance(phys_dim, dict)
                                             else phys_dim, "phys_dim") for s in self.sites}
        self.h1, self.h2 = {}, {}
        for site, mat in ({} if h1 is None else h1).items():
            self._check_site(site)
            self.h1[site] = self._matrix(mat, self.physical_dims[site])
        for (a, b), mat in ({} if h2 is None else h2).items():
            self._check_site(a)
            self._check_site(b)
            bond_axes(a, b)
            p, q = self.physical_dims[a], self.physical_dims[b]
            mat = self._matrix(mat, p * q)
            if b < a:
                mat = mat.reshape(p, q, p, q).transpose(1, 0, 3, 2).reshape(p * q, p * q)
                a, b = b, a
            self.h2[(a, b)] = self.h2.get((a, b), 0) + mat

    def _check_site(self, site):
        if site not in self.sites:
            raise ValueError("Hamiltonian site outside lattice")

    @staticmethod
    def _matrix(mat, dim):
        mat = np.asarray(mat)
        if mat.shape != (dim, dim) or not np.all(np.isfinite(mat)):
            raise ValueError("Hamiltonian matrix has incorrect shape or nonfinite entries")
        if not np.allclose(mat, mat.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("local Hamiltonian terms must be Hermitian")
        return mat.copy()

    @classmethod
    def from_model(cls, model, shape):
        """Map ``model.basis`` row-major onto the lattice.

        Uses BasisSet.op_mat and Op.split_elementary, preserving same-site
        operator order and complex factors. Only terms supported on at most
        two neighboring basis sites are accepted. Fermionic signs must already
        be represented explicitly; this is a bosonic/spin PEPS implementation.
        """
        shape = lattice_shape(shape)
        coords = sites(shape)
        if len(model.basis) != len(coords):
            raise ValueError("model basis count does not match lattice")
        h1, h2 = {}, {}
        for term in model.ham_terms:
            elements, factor = term.split_elementary(model.dof_to_siteidx)
            local = {}
            for op in elements:
                i = model.dof_to_siteidx[op.dofs[0]]
                local[i] = model.basis[i].op_mat(op)
            indices = sorted(local)
            if len(indices) == 1:
                s = coords[indices[0]]
                h1[s] = h1.get(s, 0) + factor * local[indices[0]]
            elif len(indices) == 2:
                a, b = (coords[i] for i in indices)
                bond_axes(a, b)
                h2[(a, b)] = h2.get((a, b), 0) + factor * np.kron(*(local[i] for i in indices))
            else:
                raise ValueError("PEPS evolution supports only one- and two-site terms")
        return cls(shape, h1, h2, dict(zip(coords, model.pbond_list)))

    @classmethod
    def transverse_field_ising(cls, shape, j=1.0, h=3.044):
        """H = -j sum_<ij> Z_i Z_j - h sum_i X_i (Pauli convention)."""
        shape = lattice_shape(shape)
        x = np.array([[0., 1.], [1., 0.]])
        z = np.diag([1., -1.])
        return cls(shape, {s: -h * x for s in sites(shape)},
                   {edge: -j * np.kron(z, z) for edge in bonds(shape)})

    def terms(self):
        """Deterministic order used for Trotter sweeps."""
        return [((s,), self.h1[s]) for s in sorted(self.h1)] + [
            (edge, self.h2[edge]) for edge in sorted(self.h2)]

    def check_state(self, state):
        if state.shape != self.shape or state.physical_dims != self.physical_dims:
            raise ValueError("state and Hamiltonian lattices/physical dimensions differ")

    def energy(self, state, chi=None, cutoff=0.0):
        """Normalized total energy, using boundary contractions."""
        self.check_state(state)
        return sum((state.expectation(mat, where, chi, cutoff)
                    for where, mat in self.terms()), 0j)

    def to_sparse(self, max_size=4096):
        """Exact sparse Hamiltonian for small-system verification only."""
        dims = [self.physical_dims[s] for s in self.sites]
        size = 1
        for d in dims:
            size *= d
        if size > positive_int(max_size, "max_size"):
            raise ValueError("exact Hamiltonian exceeds max_size")
        strides = [int(np.prod(dims[i + 1:], dtype=np.int64)) for i in range(len(dims))]
        cols = np.arange(size)
        result = sparse.csr_matrix((size, size), dtype=complex)
        for where, mat in self.terms():
            indices = [self.sites.index(s) for s in where]
            local_dims = [dims[i] for i in indices]
            digits = [(cols // strides[i]) % dims[i] for i in indices]
            local_cols = np.ravel_multi_index(tuple(digits), tuple(local_dims))
            base = cols - sum(d * strides[i] for d, i in zip(digits, indices))
            for out in range(mat.shape[0]):
                values = mat[out, local_cols]
                out_digits = np.unravel_index(out, tuple(local_dims))
                rows = base + sum(d * strides[i] for d, i in zip(out_digits, indices))
                result += sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()
        result.eliminate_zeros()
        return result
