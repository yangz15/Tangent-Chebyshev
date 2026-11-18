"""Square-lattice PEPS with explicit diagonal simple-update bond weights.

Site tensors have axes (physical, up, right, down, left). Coordinates and
dense physical indices are row-major. Every internal weight is inserted once
in the wavefunction; it is NOT an exact Schmidt spectrum on a loopy lattice.
"""

from numbers import Integral

import numpy as np
import opt_einsum as oe


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def lattice_shape(shape):
    if len(shape) != 2:
        raise ValueError("shape must contain two positive integers")
    return tuple(positive_int(x, "lattice size") for x in shape)


def sites(shape):
    return tuple((r, c) for r in range(shape[0]) for c in range(shape[1]))


def bonds(shape):
    return tuple((s, t) for s in sites(shape)
                 for t in ((s[0], s[1] + 1), (s[0] + 1, s[1]))
                 if t[0] < shape[0] and t[1] < shape[1])


def bond_axes(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    try:
        return {(0, 1): (2, 4), (1, 0): (3, 1),
                (0, -1): (4, 2), (-1, 0): (1, 3)}[delta]
    except KeyError:
        raise ValueError("two-site gates require distinct nearest neighbors")


def scale_axis(tensor, vector, axis):
    shape = [1] * tensor.ndim
    shape[axis] = len(vector)
    return tensor * np.asarray(vector).reshape(shape)


def retained_rank(s, max_bond, cutoff):
    if not np.isfinite(cutoff) or cutoff < 0 or cutoff >= 1:
        raise ValueError("cutoff must be in [0, 1)")
    rank = max(1, int(np.count_nonzero(s > cutoff * s[0])))
    return rank if max_bond is None else min(rank, positive_int(max_bond, "max_bond"))


def compress_boundary(mps, chi, cutoff):
    """Left QR canonicalization, then right SVD truncation of a boundary MPS."""
    for i in range(len(mps) - 1):
        l, p, r = mps[i].shape
        q, transfer = np.linalg.qr(mps[i].reshape(l * p, r))
        mps[i] = q.reshape(l, p, q.shape[1])
        mps[i + 1] = np.tensordot(transfer, mps[i + 1], axes=(1, 0))
    for i in range(len(mps) - 1, 0, -1):
        l, p, r = mps[i].shape
        u, s, vh = np.linalg.svd(mps[i].reshape(l, p * r), full_matrices=False)
        rank = retained_rank(s, chi, cutoff)
        mps[i] = vh[:rank].reshape(rank, p, r)
        mps[i - 1] = np.tensordot(mps[i - 1], u[:, :rank] * s[:rank], axes=(2, 0))
    return mps


class Peps:
    """Finite rectangular PEPS, supporting real or complex, dense local tensors.

    ``tensors`` is a rectangular nested sequence of rank-five arrays. Boundary
    virtual legs must have dimension one. ``weights`` optionally maps ordered
    nearest-neighbor coordinate pairs to nonnegative vectors. Inputs are copied.
    No symmetry sectors, fermionic swap rules, periodic boundaries or GPU backend
    are implemented in this module.
    """

    def __init__(self, tensors, weights=None):
        self.shape = lattice_shape((len(tensors), len(tensors[0]) if len(tensors) else 0))
        if any(len(row) != self.shape[1] for row in tensors):
            raise ValueError("tensors must form a rectangular grid")
        self.sites = sites(self.shape)
        self.bonds = bonds(self.shape)
        self.tensors = {}
        for site in self.sites:
            t = np.asarray(tensors[site[0]][site[1]])
            if t.ndim != 5 or min(t.shape) < 1 or not np.all(np.isfinite(t)):
                raise ValueError("site tensors must be finite nonempty rank-five arrays")
            self.tensors[site] = np.array(t, dtype=np.result_type(t.dtype, np.float64), copy=True)
            r, c = site
            for outside, axis in ((r == 0, 1), (c == self.shape[1] - 1, 2),
                                  (r == self.shape[0] - 1, 3), (c == 0, 4)):
                if outside and t.shape[axis] != 1:
                    raise ValueError("open-boundary virtual legs must have dimension one")
        self.weights = {}
        if weights is not None and set(weights) != set(self.bonds):
            raise ValueError("weights must contain exactly the ordered lattice bonds")
        for a, b in self.bonds:
            ia, ib = bond_axes(a, b)
            d = self.tensors[a].shape[ia]
            if self.tensors[b].shape[ib] != d:
                raise ValueError("neighboring virtual dimensions do not match")
            w = np.ones(d) if weights is None else np.asarray(weights[(a, b)])
            if (w.shape != (d,) or not np.isrealobj(w) or
                    not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0)):
                raise ValueError("bond weights must be finite nonnegative nonzero vectors")
            self.weights[(a, b)] = np.array(w, dtype=float, copy=True)

    @classmethod
    def product_state(cls, shape, vectors):
        """Construct from one shared vector or a dict of vectors by coordinate."""
        shape = lattice_shape(shape)
        tensors = []
        for r in range(shape[0]):
            row = []
            for c in range(shape[1]):
                v = np.asarray(vectors[(r, c)] if isinstance(vectors, dict) else vectors)
                if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
                    raise ValueError("product vectors must be finite and nonzero")
                row.append((v / np.linalg.norm(v)).reshape(-1, 1, 1, 1, 1))
            tensors.append(row)
        return cls(tensors)

    @classmethod
    def random(cls, shape, bond_dim, phys_dim=2, seed=None, complex_valued=False):
        """Random tensors, initially unit bond weights; not a canonical gauge."""
        nr, nc = lattice_shape(shape)
        d = positive_int(bond_dim, "bond_dim")
        p = positive_int(phys_dim, "phys_dim")
        rng = np.random.RandomState(seed)
        tensors = []
        for r in range(nr):
            row = []
            for c in range(nc):
                dims = (p, d if r else 1, d if c < nc - 1 else 1,
                        d if r < nr - 1 else 1, d if c else 1)
                t = rng.normal(size=dims)
                if complex_valued:
                    t = t + 1j * rng.normal(size=dims)
                row.append(t / np.linalg.norm(t))
            tensors.append(row)
        return cls(tensors)

    @property
    def max_bond(self):
        return max((len(w) for w in self.weights.values()), default=1)

    @property
    def physical_dims(self):
        return {s: self.tensors[s].shape[0] for s in self.sites}

    def copy(self):
        return type(self)([[self.tensors[(r, c)] for c in range(self.shape[1])]
                          for r in range(self.shape[0])], self.weights)

    def _incident(self, site):
        for edge in self.bonds:
            if site in edge:
                axes = bond_axes(*edge)
                yield edge, axes[edge.index(site)]

    def absorbed_tensors(self):
        """Return ordinary PEPS tensors with sqrt(weight) on each bond end."""
        out = {}
        for site, tensor in self.tensors.items():
            tensor = tensor.copy()
            for edge, axis in self._incident(site):
                tensor = scale_axis(tensor, np.sqrt(self.weights[edge]), axis)
            out[site] = tensor
        return out

    def to_dense(self, max_size=2 ** 20):
        """Exact contraction to a row-major vector; exponential diagnostic only.

        ``max_size`` bounds the output size, not every contraction intermediate.
        """
        size = 1
        for p in self.physical_dims.values():
            size *= p
        if size > positive_int(max_size, "max_size"):
            raise ValueError("dense state exceeds max_size; use boundary contraction")
        labels = {s: [i] + [None] * 4 for i, s in enumerate(self.sites)}
        label = len(self.sites)
        for a, b in self.bonds:
            ia, ib = bond_axes(a, b)
            labels[a][ia] = labels[b][ib] = label
            label += 1
        args = []
        for site, tensor in self.absorbed_tensors().items():
            for axis in range(1, 5):
                if labels[site][axis] is None:
                    labels[site][axis] = label
                    label += 1
            args.extend((tensor, labels[site]))
        args.append(list(range(len(self.sites))))
        return oe.contract(*args, optimize="greedy").reshape(-1)

    def contract(self, operators=None, chi=None, cutoff=0.0):
        """Contract <psi|product(operators)|psi> with a row boundary MPS.

        ``operators`` maps sites to matrices (output, input). ``chi=None`` and
        ``cutoff=0`` give untruncated contraction, exponential in lattice width.
        A finite chi bounds boundary bonds after each row; compare successive
        chi values to assess convergence. No dense wavefunction is constructed.
        """
        if chi is not None:
            positive_int(chi, "chi")
        retained_rank(np.ones(1), None, cutoff)
        operators = {} if operators is None else operators
        if not set(operators).issubset(self.sites):
            raise ValueError("operator site outside lattice")
        tensors = self.absorbed_tensors()
        boundary = [np.ones((1, 1, 1)) for _ in range(self.shape[1])]
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                t = tensors[(r, c)]
                op = np.asarray(operators.get((r, c), np.eye(t.shape[0])))
                if op.shape != (t.shape[0], t.shape[0]) or not np.all(np.isfinite(op)):
                    raise ValueError("local operator has incorrect shape or nonfinite entries")
                # Pair each ket virtual leg with its bra counterpart.
                layer = oe.contract("purdl,qURDL,qp->uUrRdDlL", t, t.conj(), op,
                                    optimize="greedy")
                layer = layer.reshape(tuple(d * d for d in t.shape[1:]))
                old = boundary[c]
                merged = np.tensordot(old, layer, axes=(1, 0))  # a,b,r,d,l
                boundary[c] = merged.transpose(0, 4, 3, 1, 2).reshape(
                    old.shape[0] * layer.shape[3], layer.shape[2],
                    old.shape[2] * layer.shape[1])
            compress_boundary(boundary, chi, cutoff)
        value = np.ones((1, 1))
        for t in boundary:
            value = value @ t[:, 0, :]
        return value.item()

    def norm_squared(self, chi=None, cutoff=0.0):
        value = self.contract(chi=chi, cutoff=cutoff)
        if not np.isfinite(value) or abs(np.imag(value)) > 1e-9 * abs(value) or np.real(value) <= 0:
            raise ValueError("nonpositive or invalid norm; check state and increase chi")
        return float(np.real(value))

    def normalize(self, chi=None):
        """Normalize in place using the requested boundary contraction."""
        self.tensors[(0, 0)] /= np.sqrt(self.norm_squared(chi=chi))
        return self

    def expectation_product(self, operators, chi=None, cutoff=0.0):
        return self.contract(operators, chi, cutoff) / self.norm_squared(chi, cutoff)

    def expectation(self, operator, where, chi=None, cutoff=0.0):
        """Normalized one- or two-site expectation, in the supplied site order."""
        where = tuple(where)
        if len(where) not in (1, 2) or len(set(where)) != len(where) or any(s not in self.sites for s in where):
            raise ValueError("where must contain one or two distinct lattice coordinates")
        dims = [self.physical_dims[s] for s in where]
        operator = np.asarray(operator)
        size = int(np.prod(dims))
        if operator.shape != (size, size) or not np.all(np.isfinite(operator)):
            raise ValueError("operator must be a finite matrix with matching physical dimensions")
        if len(where) == 1:
            return self.expectation_product({where[0]: operator}, chi, cutoff)
        p, q = dims
        matrix = operator.reshape(p, q, p, q).transpose(0, 2, 1, 3).reshape(p * p, q * q)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        value = 0j
        for k in range(len(s)):
            if s[k] > np.finfo(float).eps * max(matrix.shape) * s[0]:
                ops = {where[0]: u[:, k].reshape(p, p), where[1]: vh[k].reshape(q, q)}
                value += s[k] * self.contract(ops, chi, cutoff)
        return value / self.norm_squared(chi, cutoff)

    def apply_one_site_gate(self, gate, site, renormalize=False):
        """Apply a local matrix in place. Optional rescaling discards global norm."""
        t = self.tensors[site]
        gate = np.asarray(gate)
        if gate.shape != (t.shape[0], t.shape[0]) or not np.all(np.isfinite(gate)):
            raise ValueError("gate has incorrect shape or nonfinite entries")
        out = np.tensordot(gate, t, axes=(1, 0))
        if renormalize:
            norm = np.linalg.norm(out)
            if norm == 0:
                raise ValueError("gate annihilated the state")
            out /= norm
        self.tensors[site] = out

    def apply_two_site_gate(self, gate, a, b, max_bond=None, cutoff=1e-12,
                            renormalize=False):
        """QR-reduced simple update, returning local discarded squared weight.

        The discarded weight uses the product environment of simple update;
        it is not a bound on the global PEPS error. With no truncation this
        applies the gate exactly, including on a loop. ``renormalize=True``
        discards a global scalar and is intended for imaginary-time evolution.
        """
        ia, ib = bond_axes(a, b)
        edge = tuple(sorted((a, b)))
        if edge not in self.weights:
            raise ValueError("gate sites outside lattice")
        if max_bond is not None:
            positive_int(max_bond, "max_bond")
        retained_rank(np.ones(1), None, cutoff)
        pa, pb = self.physical_dims[a], self.physical_dims[b]
        gate = np.asarray(gate)
        if gate.shape != (pa * pb, pa * pb) or not np.all(np.isfinite(gate)):
            raise ValueError("gate has incorrect shape or nonfinite entries")
        reduced = []
        for site, axis in ((a, ia), (b, ib)):
            t = self.tensors[site].copy()
            for other, leg in self._incident(site):
                if other != edge:
                    t = scale_axis(t, self.weights[other], leg)
            order = [leg for leg in range(1, 5) if leg != axis] + [0, axis]
            t = t.transpose(order)
            q, r = np.linalg.qr(t.reshape(-1, t.shape[-2] * t.shape[-1]))
            reduced.append((q, r.reshape(-1, t.shape[-2], t.shape[-1]), t.shape, order))
        qa, ra, shape_a, order_a = reduced[0]
        qb, rb, shape_b, order_b = reduced[1]
        theta = oe.contract("api,i,bqi,PQpq->aPbQ", ra, self.weights[edge], rb,
                            gate.reshape(pa, pb, pa, pb), optimize="greedy")
        u, s, vh = np.linalg.svd(theta.reshape(ra.shape[0] * pa, rb.shape[0] * pb),
                               full_matrices=False)
        if not np.all(np.isfinite(s)) or s[0] == 0:
            raise ValueError("gate annihilated the state or generated nonfinite values")
        rank = retained_rank(s, max_bond, cutoff)
        discarded = float(np.sum((s[rank:] / s[0]) ** 2) / np.sum((s / s[0]) ** 2))
        new_weights = s[:rank].copy()
        if renormalize:
            new_weights /= np.linalg.norm(new_weights)
        factors = (u[:, :rank].reshape(ra.shape[0], pa, rank),
                   vh[:rank].T.reshape(rb.shape[0], pb, rank))
        for site, q, shape, order, factor in (
                (a, qa, shape_a, order_a, factors[0]),
                (b, qb, shape_b, order_b, factors[1])):
            out = (q @ factor.reshape(q.shape[1], -1)).reshape(shape[:-1] + (rank,))
            out = out.transpose(np.argsort(order))
            for other, leg in self._incident(site):
                if other != edge:
                    w = self.weights[other]
                    inv = np.zeros_like(w)
                    np.divide(1.0, w, out=inv, where=w > 0)
                    out = scale_axis(out, inv, leg)
            self.tensors[site] = out
        self.weights[edge] = new_weights
        return discarded
