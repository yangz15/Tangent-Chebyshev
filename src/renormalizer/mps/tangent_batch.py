"""Batched, device-resident S8--S10 tangent Hamiltonian contractions.

Canonicalization and charge metadata are supplied by ``MpsTangentSpace``.
After setup, all contractions, gauge projections and source batches remain in
the chosen NumPy/CuPy backend. No dense Hilbert-space operator is constructed.
The reference, MPOs and sector must remain fixed for the lifetime of this object.
"""
from __future__ import annotations

import numpy as np


class BatchedMpsTangentSpace:
    """Leading tensor axis labels independent sources in one tangent sector."""

    def __init__(self, space, backend="numpy", offload=False):
        if backend == "cupy":
            import cupy as xp
            xp.cuda.runtime.getDeviceCount()  # Fail explicitly without CUDA.
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError("backend must be numpy or cupy")
        self.xp, self.base, self.backend = xp, space, backend
        self.offload = offload
        self.left = [xp.asarray(a) for a in space.left]
        self.right = [xp.asarray(a) for a in space.right]
        self.masks = [xp.asarray(a) for a in space.qn_masks]
        self.gauge_masks = [xp.asarray(a) for a in space.gauge_qn_masks]
        self.shapes = space.shapes
        self._cache = {}

    def host(self, a):
        return self.xp.asnumpy(a) if self.backend == "cupy" else np.asarray(a)

    def synchronize(self):
        if self.backend == "cupy":
            self.xp.cuda.get_current_stream().synchronize()

    def _left(self, env, bra, op, ket):
        from renormalizer.mps import bounded_contract
        if self.backend == "cupy" and bounded_contract.needs_environment(
                env, bra, op, ket, "L", ket_first=True):
            return bounded_contract.left(env, bra, op, ket)
        env, op = self.xp.asarray(env), self.xp.asarray(op)
        e = self.xp.einsum
        t = e("kawc,kcqd->kawqd", env, ket, optimize=True)
        t = e("kawqd,wpqv->kapvd", t, op, optimize=True)
        return e("apb,kapvd->kbvd", bra.conj(), t, optimize=True)

    def _right(self, env, bra, op, ket):
        from renormalizer.mps import bounded_contract
        if self.backend == "cupy" and bounded_contract.needs_environment(
                env, bra, op, ket, "R", ket_first=True):
            return bounded_contract.right(env, bra, op, ket)
        env, op = self.xp.asarray(env), self.xp.asarray(op)
        e = self.xp.einsum
        t = e("kbvd,kcqd->kbvcq", env, ket, optimize=True)
        t = e("kbvcq,wpqv->kbwpc", t, op, optimize=True)
        return e("apb,kbwpc->kawc", bra.conj(), t, optimize=True)

    def _effective(self, left, op, right, ket):
        from renormalizer.mps import bounded_contract
        if self.backend == "cupy" and bounded_contract.needs_effective(left, op, right, ket):
            return bounded_contract.effective(left, op, right, ket)
        left, op, right = (self.xp.asarray(a) for a in (left, op, right))
        e = self.xp.einsum
        t = e("kawc,kcqd->kawqd", left, ket, optimize=True)
        t = e("kawqd,wpqv->kapvd", t, op, optimize=True)
        return e("kapvd,kbvd->kapb", t, right, optimize=True)

    def project(self, tensors):
        if len(tensors) != len(self.shapes):
            raise ValueError("wrong number of site tensors")
        result = []
        batch = tensors[0].shape[0]
        for t, a, shape, mask, gm in zip(
                tensors, self.left, self.shapes, self.masks, self.gauge_masks):
            if t.shape != (batch, *shape):
                raise ValueError("inconsistent source batch or tangent shape")
            t = self.xp.where(mask[None], t, 0).reshape(batch, -1, shape[-1])
            a = a.reshape(-1, a.shape[-1])
            overlap = (a.conj().T @ t) * gm[None]
            t = (t - a @ overlap).reshape(batch, *shape)
            result.append(self.xp.where(mask[None], t, 0))
        return result

    def pack(self, vectors):
        if not vectors or any(v.space is not self.base for v in vectors):
            raise ValueError("need vectors from the same frozen tangent space")
        return self.project([self.xp.asarray(np.stack([v.tensors[i] for v in vectors]))
                             for i in range(len(self.shapes))])

    def gram(self, bra, ket):
        """Physical cross-source overlap; returned matrix stays on device."""
        return sum(a.reshape(a.shape[0], -1).conj() @
                   b.reshape(b.shape[0], -1).T for a, b in zip(bra, ket))

    def _operators(self, mpo):
        key = id(mpo)
        if key not in self._cache:
            if len(mpo) != len(self.shapes):
                raise ValueError("MPO length differs from reference")
            from renormalizer.mps.matrix import asnumpy
            ops = [asnumpy(t.array) if self.offload else self.xp.asarray(asnumpy(t.array)) for t in mpo]
            self._cache[key] = [mpo, ops, None]
        return self._cache[key]

    def prepare(self, mpo):
        if np.any(np.asarray(mpo.qntot) != 0):
            raise ValueError("tangent Hamiltonian must be charge neutral")
        cached = self._operators(mpo)
        if cached[2] is None:
            dtype = self.xp.result_type(*self.left, *cached[1])
            one = self.xp.ones((1, 1, 1, 1), dtype=dtype)
            if self.offload:
                one = self.host(one)
            left = [one]
            right = [None] * (len(self.shapes) + 1)
            right[-1] = one
            for a, op in zip(self.left, cached[1]):
                value = self._left(left[-1], a, op, a[None])
                left.append(self.host(value) if self.offload else value)
            for i in reversed(range(len(self.shapes))):
                b = self.right[i]
                value = self._right(right[i+1], b, cached[1][i], b[None])
                right[i] = self.host(value) if self.offload else value
            cached[2] = left, right
        return cached[1], *cached[2]

    def apply(self, mpo, tensors, energy_shift=0.0):
        """Batched P(H-energy_shift)P action with shared ordinary environments."""
        ops, left, right = self.prepare(mpo)
        n = len(self.shapes)
        il = [None]*(n+1)
        # No output consumes the insertion beyond the final site.
        for i in range(n - 1):
            value = self._left(left[i], self.left[i], ops[i], tensors[i])
            if i:
                value += self._left(il[i], self.left[i], ops[i], self.right[i][None])
            il[i+1] = self.host(value) if self.offload else value
        result, inserted_right = [None]*n, None
        # Keep only the current right insertion; consume/release left insertions.
        for i in reversed(range(n)):
            t = self._effective(left[i], ops[i], right[i+1], tensors[i])
            if i:
                t += self._effective(il[i], ops[i], right[i+1], self.right[i][None])
            if i+1 < n:
                t += self._effective(left[i], ops[i], inserted_right, self.left[i][None])
            result[i] = t - energy_shift*tensors[i]
            if i:
                value = self._right(right[i+1], self.right[i], ops[i], tensors[i])
                if inserted_right is not None:
                    value += self._right(inserted_right, self.right[i], ops[i], self.left[i][None])
                inserted_right = value
            il[i] = None
        return self.project(result)

    def sources(self, mpos):
        """Project arbitrary MPO sources, preserving exact sector selection."""
        sources = []
        for mpo in mpos:
            expected = self.base.reference_qntot + np.asarray(mpo.qntot).reshape(-1)
            if not np.array_equal(expected, self.base.target_qntot):
                raise ValueError("source MPO maps into the wrong tangent sector")
            ops = self._operators(mpo)[1]
            dtype = self.xp.result_type(*self.right, *ops)
            one = self.xp.ones((1, 1, 1, 1), dtype=dtype)
            left, right = [one], [None]*(len(ops)+1)
            right[-1] = one
            for a, b, op in zip(self.left, self.right, ops):
                left.append(self._left(left[-1], a, op, b[None]))
            for i in reversed(range(len(ops))):
                right[i] = self._right(right[i+1], self.right[i], ops[i], self.right[i][None])
            sources.append([self._effective(left[i], op, right[i+1], self.right[i][None])
                            for i, op in enumerate(ops)])
        if not sources:
            raise ValueError("at least one source is required")
        return self.project([self.xp.concatenate([s[i] for s in sources])
                             for i in range(len(self.shapes))])

    def storage_bytes(self):
        arrays = self.left + self.right + self.masks + self.gauge_masks
        for _, ops, env in self._cache.values():
            arrays += ops
            if env is not None:
                arrays += env[0] + env[1]
        return sum(a.nbytes for a in arrays)
