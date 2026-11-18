"""Charge-resolved, rectangular MPO image maps for the NESS solver.

No normal operator or projected Liouvillian is used. QR output frames retain
every allowed row; packing removes only entries forbidden by additive charges.
Small dense charge blocks are padded and batched for GPU matrix multiplication.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse

from renormalizer.mps.backend import xp, USE_GPU


def groups(labels):
    result = defaultdict(list)
    for index, label in enumerate(labels):
        result[tuple(label)].append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


def sector_prefixes(physical, total):
    """Reachable charges, without enumerating physical configurations."""
    zero = (0,)*len(total)
    prefix, suffix = [{zero}], [{zero}]
    bounded = all(np.all(q >= 0) for q in physical) and np.all(np.asarray(total) >= 0)
    def advance(previous, labels):
        result = {tuple(np.add(a, b)) for a in previous for b in groups(labels)}
        if bounded:
            result = {q for q in result if all(a <= b for a, b in zip(q, total))}
        return result
    for labels in physical:
        prefix.append(advance(prefix[-1], labels))
    for labels in physical[::-1]:
        suffix.append(advance(suffix[-1], labels))
    return [p & {tuple(np.subtract(total, q)) for q in s}
            for p, s in zip(prefix, suffix[::-1])]


@dataclass
class Image:
    data: object
    charges: np.ndarray


def image_step(environment, operator, state, bond_charges, mpo_charges,
               physical, allowed, to_right):
    """Untruncated QR in each output charge, never forming the full image core."""
    labels = (mpo_charges[:, None, :]+bond_charges[None, :, :]).reshape(-1, physical.shape[1])
    columns = groups(labels)
    frame = groups(environment.charges)
    blocks, new_charges = [], []
    width, bond = len(mpo_charges), len(bond_charges)
    for charge, indices in columns.items():
        if charge not in allowed:
            continue
        channels, bonds = indices//bond, indices % bond
        c, b = xp.asarray(channels), xp.asarray(bonds)
        rows = []
        for p, qp in enumerate(physical):
            key = tuple(np.subtract(charge, qp) if to_right else np.add(charge, qp))
            selected = frame.get(key)
            if selected is None:
                continue
            if to_right:
                # T[a,i,c] = sum_q W[a,p,q,b_c] A[i,q,j_c].
                w = operator[:, p, :, :][:, :, c]
                a = state[:, :, b]
            else:
                # T[b,j,c] = sum_q W[a_c,p,q,b] A[i_c,q,j].
                w = operator[c, p, :, :].transpose(2, 1, 0)
                a = state[b, :, :].transpose(2, 1, 0)
            t = xp.einsum("aqc,iqc->aic", w, a)
            e = environment.data[xp.asarray(selected)].reshape(len(selected), -1)
            rows.append(e @ t.reshape(-1, len(indices)))
        if not rows:
            continue
        block = xp.concatenate(rows, axis=0)
        # The right frame is conjugate-orthonormal (QR of the adjoint).
        _, r = xp.linalg.qr(block if to_right else block.conj(), mode="reduced")
        if not to_right:
            r = r.conj()
        blocks.append((indices, r))
        new_charges.extend([charge]*r.shape[0])
    data = xp.zeros((len(new_charges), width*bond), dtype=xp.complex128)
    offset = 0
    for indices, r in blocks:
        data[offset:offset+len(r), xp.asarray(indices)] = r
        offset += len(r)
    return Image(data.reshape(-1, width, bond), np.asarray(new_charges, dtype=int))


def image_norm(operator, state, charges, mpo_charges, physical, allowed):
    image = Image(xp.ones((1, 1, 1), dtype=xp.complex128), np.zeros((1, physical[0].shape[1]), int))
    for j, (w, a) in enumerate(zip(operator, state)):
        image = image_step(image, w, a, charges[j+1], mpo_charges[j+1],
                           physical[j], allowed[j+1], True)
    return float(xp.linalg.norm(image.data.ravel()))


class Layout:
    """Packed local tensor, with groups ordered (physical, left, right)."""
    def __init__(self, left, physical, right):
        self.left, self.right = groups(left), groups(right)
        labels = physical[0]
        for following in physical[1:]:
            labels = (labels[:, None, :]+following[None, :, :]).reshape(-1, labels.shape[1])
        self.physical = groups(labels)
        self.shape = (len(left),)+tuple(len(p) for p in physical)+(len(right),)
        self.blocks, self.size = {}, 0
        for ql, li in self.left.items():
            for qr, ri in self.right.items():
                delta = tuple(np.subtract(qr, ql))
                pi = self.physical.get(delta)
                if pi is None:
                    continue
                shape = (len(pi), len(li), len(ri))
                self.blocks[ql, qr] = (self.size, shape, li, pi, ri)
                self.size += math.prod(shape)
        self._indices = None

    def indices(self):
        if self._indices is None:
            indices = []
            for _, _, li, pi, ri in self.blocks.values():
                ix = ((li[None, :, None]*math.prod(self.shape[1:-1])+pi[:, None, None])
                      *self.shape[-1]+ri[None, None, :])
                indices.append(ix.ravel())
            self._indices = xp.asarray(np.concatenate(indices))
        return self._indices

    def pack(self, tensor):
        return tensor.ravel()[self.indices()]

    def unpack(self, vector):
        result = xp.zeros(self.shape, dtype=xp.complex128)
        result.ravel()[self.indices()] = vector
        return result


class PhysicalMap:
    """Cached sparse physical operators for each pair of MPO boundary channels."""
    def __init__(self, operators):
        first = operators[0]
        self.channels = []
        if len(operators) == 1:
            for a in range(first.shape[0]):
                for b in range(first.shape[-1]):
                    s = sparse.csr_matrix(first[a, :, :, b])
                    if s.nnz:
                        self.channels.append((a, b, s))
        else:
            second = operators[1]
            left, right = defaultdict(list), defaultdict(list)
            for a in range(first.shape[0]):
                for c in range(first.shape[-1]):
                    if np.any(first[a, :, :, c]):
                        left[c].append((a, sparse.csr_matrix(first[a, :, :, c])))
            for c in range(second.shape[0]):
                for b in range(second.shape[-1]):
                    if np.any(second[c, :, :, b]):
                        right[c].append((b, sparse.csr_matrix(second[c, :, :, b])))
            terms = {}
            for c in left.keys() & right.keys():
                for a, l in left[c]:
                    for b, r in right[c]:
                        term = sparse.kron(l, r, format="csr")
                        terms[a, b] = terms.get((a, b), 0)+term
            for (a, b), s in terms.items():
                s.eliminate_zeros()
                if s.nnz:
                    self.channels.append((a, b, s))
        self.cache = {}

    def block(self, channel, pin, pout):
        key = (channel, tuple(pin), tuple(pout))
        if key not in self.cache:
            s = self.channels[channel][2][pout][:, pin].toarray()
            self.cache[key] = s if np.any(s) else None
        return self.cache[key]


_gather_kernel = _scatter_kernel = None


def _kernels():
    global _gather_kernel, _scatter_kernel
    if _gather_kernel is not None:
        return _gather_kernel, _scatter_kernel
    source = r'''
    extern "C" __global__ void gather(const double2* x, double2* y,
       const long long* meta, long long size, int F, int L, int R) {
       long long idx = (long long)blockDim.x*blockIdx.x+threadIdx.x;
       if (idx >= size) return;
       int r=idx%R, l=(idx/R)%L, f=(idx/(R*L))%F;
       long long b=idx/((long long)R*L*F);
       const long long* m=meta+4*b;
       double2 value=make_double2(0.,0.);
       if(f<m[1] && l<m[2] && r<m[3])
          value=x[m[0]+(f*m[2]+l)*m[3]+r];
       y[idx]=value;
    }
    extern "C" __global__ void scatter(const double2* x, double2* y,
       const long long* meta, long long size, int F, int L, int R) {
       long long idx = (long long)blockDim.x*blockIdx.x+threadIdx.x;
       if (idx >= size) return;
       int r=idx%R, l=(idx/R)%L, f=(idx/(R*L))%F;
       long long b=idx/((long long)R*L*F);
       const long long* m=meta+4*b;
       if(f<m[1] && l<m[2] && r<m[3]) {
          double* dst=(double*)(y+m[0]+(f*m[2]+l)*m[3]+r);
          atomicAdd(dst,x[idx].x); atomicAdd(dst+1,x[idx].y);
       }
    }'''
    _gather_kernel = xp.RawKernel(source, "gather")
    _scatter_kernel = xp.RawKernel(source, "scatter")
    return _gather_kernel, _scatter_kernel


def _gather(vector, metadata, shape):
    allocate = xp.empty if USE_GPU else xp.zeros
    result = allocate((len(metadata),)+shape, dtype=xp.complex128)
    if USE_GPU:
        gather, _ = _kernels()
        gather(((result.size+255)//256,), (256,),
               (vector, result, metadata, np.int64(result.size), *(np.int32(v) for v in shape)))
    else:
        for dst, (offset, f, l, r) in zip(result, metadata):
            dst[:f, :l, :r] = vector[offset:offset+f*l*r].reshape(f, l, r)
    return result


def _scatter(values, vector, metadata):
    if USE_GPU:
        _, scatter = _kernels()
        values = xp.ascontiguousarray(values)
        scatter(((values.size+255)//256,), (256,),
                (values, vector, metadata, np.int64(values.size),
                 *(np.int32(v) for v in values.shape[1:])))
    else:
        for src, (offset, f, l, r) in zip(values, metadata):
            vector[offset:offset+f*l*r] += src[:f, :l, :r].ravel()


def _power_two(n):
    return 1 << (n-1).bit_length()


class BlockLocalMap:
    def __init__(self, left, physical_map, right, input_layout, output_layout,
                 left_mpo_charges, right_mpo_charges, config):
        self.input, self.output, self.matvecs = input_layout, output_layout, 0
        if max(self.input.size, self.output.size) > config.max_local_elements:
            raise MemoryError("Packed local LSQR tensor exceeds max_local_elements")
        buckets = defaultdict(list)
        for (ql, qr), src in self.input.blocks.items():
            for channel, (a, b, _) in enumerate(physical_map.channels):
                key = (tuple(np.add(ql, left_mpo_charges[a])),
                       tuple(np.add(qr, right_mpo_charges[b])))
                dst = self.output.blocks.get(key)
                if dst is None:
                    continue
                s = physical_map.block(channel, src[3], dst[3])
                if s is None:
                    continue
                shape = tuple(_power_two(int(n)) for n in src[1]+dst[1])
                buckets[shape].append((src, dst, a, b, s))
        self.batches = []
        for shape, records in buckets.items():
            f, i, j, g, k, l = shape
            # Bound padded intermediates in both forward and adjoint directions.
            largest = max(f*i*j, f*k*j, f*k*l, g*k*l, g*i*l, g*i*j)
            chunk = max(1, config.block_workspace_elements//largest)
            for start in range(0, len(records), chunk):
                subset = records[start:start+chunk]
                sc = np.zeros((len(subset), g, f), np.complex128)
                # Gather all environment coefficients in one device operation.
                lix = np.zeros((len(subset), k, i), np.int64)
                rix = np.zeros((len(subset), l, j), np.int64)
                lm, rm = np.zeros(lix.shape, bool), np.zeros(rix.shape, bool)
                im, om = [], []
                for z, (src, dst, a, b, s) in enumerate(subset):
                    _, (fi, ii, ji), li, _, ri = src
                    _, (go, ko, lo), lk, _, rl = dst
                    lix[z, :ko, :ii] = ((lk[:, None]*left.data.shape[1]+a)
                                       *left.data.shape[2]+li[None, :])
                    rix[z, :lo, :ji] = ((rl[:, None]*right.data.shape[1]+b)
                                       *right.data.shape[2]+ri[None, :])
                    lm[z, :ko, :ii], rm[z, :lo, :ji] = True, True
                    sc[z, :go, :fi] = s
                    im.append((src[0], fi, ii, ji))
                    om.append((dst[0], go, ko, lo))
                lc = left.data.ravel()[xp.asarray(lix)]*xp.asarray(lm)
                rc = right.data.ravel()[xp.asarray(rix)]*xp.asarray(rm)
                self.batches.append((lc, rc, xp.asarray(sc),
                                     xp.asarray(im, dtype=xp.int64), xp.asarray(om, dtype=xp.int64), shape))
        self.coefficient_bytes = sum(sum(x.nbytes for x in batch[:5]) for batch in self.batches)
        self.block_count = sum(len(batch[3]) for batch in self.batches)

    def _apply(self, vector, adjoint):
        self.matvecs += 1
        result = xp.zeros(self.input.size if adjoint else self.output.size, dtype=xp.complex128)
        for lc, rc, sc, im, om, shape in self.batches:
            f, i, j, g, k, l = shape
            if adjoint:
                lc, rc, sc = (a.conj().swapaxes(1, 2) for a in (lc, rc, sc))
                im, om = om, im
                f, i, j, g, k, l = g, k, l, f, i, j
            x = _gather(vector, im, (f, i, j))
            y = xp.matmul(lc[:, None], x)
            y = xp.matmul(y, rc.swapaxes(1, 2)[:, None])
            y = xp.matmul(sc, y.reshape(len(sc), f, k*l)).reshape(len(sc), g, k, l)
            _scatter(y, result, om)
        return result

    def forward(self, vector):
        return self._apply(vector, False)

    def adjoint(self, vector):
        return self._apply(vector, True)
