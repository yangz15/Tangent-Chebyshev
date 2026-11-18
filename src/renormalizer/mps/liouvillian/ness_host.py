"""Host-backed rectangular LSQR for large charge-conserving NESS MPS.

Keeps the original B=L V objective and complex LSQR. Dense native MPS cores,
batched coefficients and output Krylov vectors live in host RAM; contractions,
input Krylov vectors and charge-block QR/SVD run on the selected backend.
No dense two-site center, normal equations, or global state is constructed.
"""
from collections import defaultdict
import math
import time

import numpy as np
from opt_einsum import contract

from renormalizer import Mps
from renormalizer.mps.backend import xp, USE_GPU
from renormalizer.mps.matrix import asnumpy
from .ness_blocks import Layout, PhysicalMap, groups, sector_prefixes, _power_two, _gather, _scatter
from .tensor import compress, inner


def einsum(expr, *args):
    return contract(expr, *args, optimize="greedy")


class PackedImage:
    def __init__(self, pieces, width, bond, charge_size):
        self.width, self.bond, self.info = width, bond, {}
        arrays, labels, offset = [], [], 0
        for charge, columns, matrix in pieces:
            lookup = np.full(width*bond, -1, np.int64)
            lookup[columns] = np.arange(len(columns))
            self.info[charge] = (offset, matrix.shape, columns, lookup)
            arrays.append(matrix.ravel())
            labels.extend([charge]*len(matrix))
            offset += matrix.size
        self.data = xp.concatenate(arrays) if arrays else xp.empty(0, dtype=xp.complex128)
        self.charges = np.asarray(labels, dtype=int).reshape(-1, charge_size)

    @classmethod
    def boundary(cls, charge):
        return cls([(tuple(charge), np.array([0]), xp.ones((1, 1), complex))], 1, 1, len(charge))

    def block(self, charge):
        offset, shape, columns, _ = self.info[charge]
        return self.data[offset:offset+math.prod(shape)].reshape(shape), columns

    def coefficient_indices(self, charge, channel, states):
        offset, (rows, cols), _, lookup = self.info[charge]
        selected = lookup[channel*self.bond+states]
        if np.any(selected < 0):
            raise ValueError("Inconsistent charge metadata in MPO image")
        return offset+np.arange(rows)[:, None]*cols+selected[None, :]

    def on_device(self):
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.data = xp.asarray(self.data)
        return result

    def offload(self):
        self.data = asnumpy(self.data)
        return self

    def to_dense(self):
        """Small-reference/testing conversion; production contractions stay packed."""
        result = xp.zeros((len(self.charges), self.width*self.bond), complex)
        row = 0
        for charge in self.info:
            matrix, columns = self.block(charge)
            result[row:row+len(matrix), xp.asarray(columns)] = matrix
            row += len(matrix)
        return result.reshape(-1, self.width, self.bond)


def image_step(environment, operator, state, bond_charges, mpo_charges,
               physical, allowed, to_right, workspace_elements=2_000_000, qr_max_elements=16_000_000):
    """Exact QR of each image charge, with packed incoming/outgoing environments."""
    labels = (mpo_charges[:, None]+bond_charges[None, :]).reshape(-1, physical.shape[1])
    pieces = []
    q = xp.arange(operator.shape[2])[None, :, None]
    for charge, columns in groups(labels).items():
        if charge not in allowed:
            continue
        new_channel = xp.asarray(columns//len(bond_charges))[None, None, :]
        new_state = xp.asarray(columns % len(bond_charges))[None, None, :]
        rows = []
        for p, qp in enumerate(physical):
            key = tuple(np.subtract(charge, qp) if to_right else np.add(charge, qp))
            if key not in environment.info:
                continue
            e, old_columns = environment.block(key)
            # Split the summed image columns before physical gathers can grow
            # as (MPO*chi)**2*p in a heavily populated charge sector.
            chunk = max(1, workspace_elements//(operator.shape[2]*len(columns)))
            row = xp.zeros((e.shape[0], len(columns)), complex)
            for start in range(0, len(old_columns), chunk):
                selected = old_columns[start:start+chunk]
                old_channel = xp.asarray(selected//environment.bond)[:, None, None]
                old_state = xp.asarray(selected % environment.bond)[:, None, None]
                if to_right:
                    w = operator[old_channel, p, q, new_channel]
                    a = state[old_state, q, new_state]
                else:
                    w = operator[new_channel, p, q, old_channel]
                    a = state[new_state, q, old_state]
                row += e[:, start:start+chunk] @ xp.sum(w*a, axis=1)
            rows.append(row)
            del w, a, row
        if not rows:
            continue
        block = xp.concatenate(rows, axis=0)
        del rows
        # Only R is needed. Particularly large individual charge blocks use
        # host LAPACK rather than exhausting GPU QR workspace.
        if USE_GPU and block.size > qr_max_elements:
            host = asnumpy(block)
            r = xp.asarray(np.linalg.qr(host if to_right else host.conj(), mode="r"))
            del host
        else:
            r = xp.linalg.qr(block if to_right else block.conj(), mode="r")
        pieces.append((charge, columns, r if to_right else r.conj()))
        del block, r
    return PackedImage(pieces, len(mpo_charges), len(bond_charges), physical.shape[1])


def image_norm(operator, state, charges, mpo_charges, physical, allowed):
    image = PackedImage.boundary(charges[0][0])
    for j, (w, a) in enumerate(zip(operator, state)):
        image = image_step(image, w, xp.asarray(a), charges[j+1], mpo_charges[j+1],
                           physical[j], allowed[j+1], True)
    value = float(xp.linalg.norm(image.data))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite streamed tensor norm")
    return value


def packed_center(layout, left, right=None):
    result = xp.empty(layout.size, dtype=xp.complex128)
    if right is None:
        return layout.pack(left)
    for offset, shape, li, pi, ri in layout.blocks.values():
        p, q = pi//right.shape[1], pi % right.shape[1]
        a = left[xp.asarray(li)[:, None], xp.asarray(p)[None, :], :].transpose(1, 0, 2)
        b = right[:, xp.asarray(q)[:, None], xp.asarray(ri)[None, :]].transpose(1, 0, 2)
        result[offset:offset+math.prod(shape)] = xp.matmul(a, b).ravel()
    return result


def packed_functional(layout, lt, trace, rt):
    if len(trace) == 1:
        middle = trace[0].conj().transpose(1, 0, 2)
    else:
        middle = einsum("apc,cqb->pqab", trace[0].conj(), trace[1].conj())
        middle = middle.reshape(-1, middle.shape[-2], middle.shape[-1])
    result = xp.empty(layout.size, dtype=xp.complex128)
    for offset, shape, li, pi, ri in layout.blocks.values():
        block = einsum("ai,fab,bj->fij", lt[:, xp.asarray(li)], middle[xp.asarray(pi)], rt[:, xp.asarray(ri)])
        result[offset:offset+math.prod(shape)] = block.ravel()
    return result


def split_packed(vector, layout, charges_left, charges_right, physical, to_right, config):
    """SVD permutations scale with packed size, never with chi**2*p**2."""
    p0, p1 = map(len, physical)
    row_labels = (charges_left[:, None]+physical[0][None, :]).reshape(-1, physical[0].shape[1])
    col_labels = (charges_right[None, :]-physical[1][:, None]).reshape(-1, physical[0].shape[1])
    row_groups, col_groups = groups(row_labels), groups(col_labels)
    row_positions, col_positions, permutations = {}, {}, {}
    for charge in row_groups.keys() & col_groups.keys():
        rows, cols = row_groups[charge], col_groups[charge]
        rr, cc = np.full(len(row_labels), -1, int), np.full(len(col_labels), -1, int)
        rr[rows], cc[cols] = np.arange(len(rows)), np.arange(len(cols))
        row_positions[charge], col_positions[charge] = rr, cc
        permutations[charge] = np.full((len(rows), len(cols)), -1, np.int64)
    for (ql, _), (offset, shape, li, pi, ri) in layout.blocks.items():
        for f, pair in enumerate(pi):
            a, b = divmod(int(pair), p1)
            charge = tuple(np.add(ql, physical[0][a]))
            rows = row_positions[charge][li*p0+a]
            cols = col_positions[charge][b*len(charges_right)+ri]
            indices = offset+f*len(li)*len(ri)+np.arange(len(li)*len(ri)).reshape(len(li), len(ri))
            permutations[charge][rows[:, None], cols[None, :]] = indices
    factors, values = [], []
    for charge in sorted(permutations):
        permutation = permutations[charge]
        if np.any(permutation < 0):
            raise ValueError("Incomplete charge-block center permutation")
        u, s, vh = xp.linalg.svd(vector[xp.asarray(permutation)], full_matrices=False)
        factors.append((charge, u, s, vh, len(values)))
        values.extend(asnumpy(s).tolist())
    values = np.asarray(values)
    total = float(np.linalg.norm(values))
    if not total or not np.isfinite(total):
        raise FloatingPointError("Local solve lost the normalized state")
    order = np.argsort(-values, kind="stable")
    rank = max(1, min(config.bond_cap, int(np.count_nonzero(values > config.cutoff*total))))
    positions = {int(old): new for new, old in enumerate(order[:rank])}
    left = np.zeros((len(row_labels), rank), complex)
    right = np.zeros((rank, len(col_labels)), complex)
    labels = np.empty((rank, physical[0].shape[1]), int)
    for charge, u, s, vh, offset in factors:
        selected = [k for k in range(len(s)) if offset+k in positions]
        if not selected:
            continue
        dst = np.array([positions[offset+k] for k in selected])
        src = xp.asarray(selected)
        rows, cols = row_groups[charge], col_groups[charge]
        left[rows[:, None], dst[None, :]] = asnumpy(u[:, src]*(1 if to_right else s[src]))
        right[dst[:, None], cols[None, :]] = asnumpy(vh[src]*(s[src, None] if to_right else 1))
        labels[dst] = charge
    return (left.reshape(len(charges_left), p0, rank),
            right.reshape(rank, p1, len(charges_right)), labels,
            float(np.linalg.norm(values[order[rank:]])/total))


class StreamLocalMap:
    """GPU contractions in tiles of complete charge blocks; host-backed output."""
    def __init__(self, left, physical_map, right, input_layout, output_layout,
                 left_mpo_charges, right_mpo_charges, config):
        self.input, self.output, self.matvecs = input_layout, output_layout, 0
        if self.input.size > config.max_local_elements:
            raise MemoryError("Packed local input exceeds max_local_elements")
        if self.output.size > config.max_host_output_elements:
            raise MemoryError("Local output exceeds max_host_output_elements")
        self.tiles, tile_for_key = [], {}
        for key, (offset, shape, *_) in self.output.blocks.items():
            size = math.prod(shape)
            if not self.tiles or self.tiles[-1][1]+size > config.host_tile_elements:
                self.tiles.append([offset, 0])
            self.tiles[-1][1] += size
            tile_for_key[key] = len(self.tiles)-1
        buckets = defaultdict(list)
        for (ql, qr), src in self.input.blocks.items():
            for channel, (a, b, _) in enumerate(physical_map.channels):
                key = (tuple(np.add(ql, left_mpo_charges[a])), tuple(np.add(qr, right_mpo_charges[b])))
                dst = self.output.blocks.get(key)
                if dst is None:
                    continue
                s = physical_map.block(channel, src[3], dst[3])
                if s is None:
                    continue
                shape = tuple(_power_two(int(n)) for n in src[1]+dst[1])
                buckets[tile_for_key[key], shape].append((src, dst, a, b, s, key))
        self.plans = defaultdict(list)
        self.coefficient_bytes = self.block_count = 0
        for (tile, shape), records in buckets.items():
            f, i, j, g, k, l = shape
            largest = max(f*i*j, f*k*j, f*k*l, g*k*l, g*i*l, g*i*j)
            chunk = max(1, config.block_workspace_elements//largest)
            for start in range(0, len(records), chunk):
                subset = records[start:start+chunk]
                sc = np.zeros((len(subset), g, f), complex)
                lix, rix = np.zeros((len(subset), k, i), np.int64), np.zeros((len(subset), l, j), np.int64)
                lm, rm = np.zeros(lix.shape, bool), np.zeros(rix.shape, bool)
                im, om = [], []
                for z, (src, dst, a, b, s, key) in enumerate(subset):
                    _, (fi, ii, ji), li, _, ri = src
                    _, (go, ko, lo), _, _, _ = dst
                    lix[z, :ko, :ii] = left.coefficient_indices(key[0], a, li)
                    rix[z, :lo, :ji] = right.coefficient_indices(key[1], b, ri)
                    lm[z, :ko, :ii], rm[z, :lo, :ji] = True, True
                    sc[z, :go, :fi] = s
                    im.append((src[0], fi, ii, ji))
                    om.append((dst[0]-self.tiles[tile][0], go, ko, lo))
                lc = asnumpy(left.data[xp.asarray(lix)]*xp.asarray(lm))
                rc = asnumpy(right.data[xp.asarray(rix)]*xp.asarray(rm))
                batch = (lc, rc, sc, np.asarray(im, np.int64), np.asarray(om, np.int64), shape)
                self.coefficient_bytes += sum(v.nbytes for v in batch[:5])
                if self.coefficient_bytes > config.max_host_coefficient_bytes:
                    raise MemoryError("Local map exceeds max_host_coefficient_bytes")
                self.block_count += len(subset)
                self.plans[tile].append(batch)
        self.batch_count = sum(map(len, self.plans.values()))

    def _block(self, vector, key, adjoint, result=None):
        if result is None:
            result = xp.zeros(self.tiles[key][1], complex)
        for batch in self.plans[key]:
            lc, rc, sc, im, om = (xp.asarray(x) for x in batch[:5])
            f, i, j, g, k, l = batch[5]
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
        self.matvecs += 1
        output = np.zeros(self.output.size, complex)
        for key in self.plans:
            offset, size = self.tiles[key]
            output[offset:offset+size] = asnumpy(self._block(vector, key, False))
        return output

    def adjoint(self, vector):
        self.matvecs += 1
        result = xp.zeros(self.input.size, complex)
        for key in self.plans:
            offset, size = self.tiles[key]
            self._block(xp.asarray(vector[offset:offset+size]), key, True, result)
        return result

    def residual_norm(self, vector):
        self.matvecs += 1
        squares = []
        for key in self.plans:
            block = self._block(vector, key, False)
            squares.append(float(xp.vdot(block, block).real))
        return math.sqrt(math.fsum(squares))


def host_norm(vector):
    # Bound each BLAS reduction even when the output exceeds 32-bit BLAS lengths.
    chunk = 16_000_000
    return math.sqrt(math.fsum(float(np.vdot(vector[i:i+chunk], vector[i:i+chunk]).real)
                               for i in range(0, vector.size, chunk)))


def lsqr_host(forward, adjoint, rhs_factory, size, tolerance, maxiter):
    """Same complex LSQR recurrence, with only the output vectors in host RAM."""
    x = xp.zeros(size, complex)
    u = rhs_factory()
    beta = host_norm(u)
    if beta == 0:
        return x, 0, 0.
    u /= beta
    v = adjoint(u)
    alpha = float(xp.linalg.norm(v))
    if alpha == 0:
        return x, 0, beta
    v /= alpha
    w = v.copy()
    rhobar, phibar, bnorm, anorm2 = alpha, beta, beta, alpha*alpha
    for iteration in range(1, maxiter+1):
        following = forward(v)
        u *= alpha
        following -= u
        u = following
        del following
        beta = host_norm(u)
        if beta:
            u /= beta
        v = adjoint(u)-beta*v
        alpha = float(xp.linalg.norm(v))
        if alpha:
            v /= alpha
        anorm2 += beta*beta+alpha*alpha
        rho = math.hypot(rhobar, beta)
        if rho == 0:
            break
        c, s = rhobar/rho, beta/rho
        theta, rhobar = s*alpha, -c*alpha
        phi, phibar = c*phibar, s*phibar
        x += (phi/rho)*w
        w = v-(theta/rho)*w
        if (abs(phibar) <= tolerance*bnorm or
                alpha*abs(s*phi) <= tolerance*math.sqrt(anorm2)*abs(phibar) or alpha == 0):
            break
    return x, iteration, bnorm


def half_sweep(state, charges, physical, operator, trace, direction, config,
               mpo_charges, allowed, physical_maps, local_callback=None):
    from .ness import _overlap_left, _overlap_right
    n = len(state)
    def step(image, j, to_right):
        bond = j+1 if to_right else j
        return image_step(image.on_device(), operator[j], xp.asarray(state[j]), charges[bond], mpo_charges[bond],
                          physical[j], allowed[bond], to_right)
    images, overlaps = [None]*(n+1), [None]*(n+1)
    if direction:
        images[n], overlaps[n] = PackedImage.boundary(charges[-1][0]), xp.ones((1, 1), complex)
        for j in range(n-1, 1, -1):
            images[j] = step(images[j+1], j, False).offload()
            overlaps[j] = _overlap_right(overlaps[j+1], trace[j], xp.asarray(state[j]))
        moving_image, moving_overlap = PackedImage.boundary(charges[0][0]), xp.ones((1, 1), complex)
        sites = range(max(1, n-1))
    else:
        images[0], overlaps[0] = PackedImage.boundary(charges[0][0]), xp.ones((1, 1), complex)
        for j in range(max(0, n-2)):
            images[j+1] = step(images[j], j, True).offload()
            overlaps[j+1] = _overlap_left(overlaps[j], trace[j], xp.asarray(state[j]))
        moving_image, moving_overlap = PackedImage.boundary(charges[-1][0]), xp.ones((1, 1), complex)
        sites = range(n-2, -1, -1) if n > 1 else [0]
    stats = dict(local_iterations=0, matvecs=0, discarded=0., trace_correction=0.,
                 max_packed_input=0, max_packed_output=0, max_coefficient_bytes=0)
    for j in sites:
        started = time.perf_counter()
        end = min(j+2, n)
        le, re = (moving_image, images[end].on_device()) if direction else (images[j].on_device(), moving_image)
        lt, rt = (moving_overlap, overlaps[end]) if direction else (overlaps[j], moving_overlap)
        layout = Layout(charges[j], physical[j:end], charges[end])
        out = Layout(le.charges, physical[j:end], re.charges)
        action = StreamLocalMap(le, physical_maps[j], re, layout, out, mpo_charges[j], mpo_charges[end], config)
        a = packed_center(layout, xp.asarray(state[j]), xp.asarray(state[j+1]) if end-j == 2 else None)
        t = packed_functional(layout, lt, trace[j:end], rt).conj()
        denominator = float(xp.vdot(t, t).real)
        if denominator < np.finfo(float).tiny:
            raise FloatingPointError("Local frame cannot represent nonzero trace")
        a += t*((1-xp.vdot(t, a))/denominator)
        def project(v):
            return v-t*(xp.vdot(t, v)/denominator)
        setup_seconds = time.perf_counter()-started
        def rhs():
            value = action.forward(a)
            value *= -1
            return value
        delta, iterations, before = lsqr_host(lambda v: action.forward(project(v)),
                                              lambda v: project(action.adjoint(v)), rhs,
                                              a.size, config.local_tolerance, config.local_maxiter)
        candidate = a+project(delta)
        if action.residual_norm(candidate) <= before*(1+1e-12):
            a = candidate
        a += t*((1-xp.vdot(t, a))/denominator)
        correction = discarded = 0.
        if end-j == 1:
            state[j] = asnumpy(layout.unpack(a))
        else:
            left, right, labels, discarded = split_packed(a, layout, charges[j], charges[end],
                                                         physical[j:end], direction, config)
            if direction:
                updated = _overlap_left(lt, trace[j], xp.asarray(left))
                center_layout = Layout(labels, physical[j+1:end], charges[end])
                covector = packed_functional(center_layout, updated, trace[j+1:end], rt)
                center = xp.asarray(right)
            else:
                updated = _overlap_right(rt, trace[j+1], xp.asarray(right))
                center_layout = Layout(charges[j], physical[j:j+1], labels)
                covector = packed_functional(center_layout, lt, trace[j:j+1], updated)
                center = xp.asarray(left)
            value = center_layout.pack(center)
            norm = float(xp.vdot(covector, covector).real)
            if norm < np.finfo(float).tiny:
                raise FloatingPointError("Compression removed all trace support; increase bond_cap")
            repair = covector.conj()*((1-xp.sum(covector*value))/norm)
            center.ravel()[center_layout.indices()] += repair
            correction = float(xp.linalg.norm(repair))
            if direction:
                right = asnumpy(center)
            else:
                left = asnumpy(center)
            state[j], state[j+1], charges[j+1] = left, right, labels
            del center, covector, value, repair, center_layout
        stats["local_iterations"] += iterations
        stats["matvecs"] += action.matvecs
        for key, value in (("discarded", discarded), ("trace_correction", correction),
                           ("max_packed_input", layout.size), ("max_packed_output", out.size),
                           ("max_coefficient_bytes", action.coefficient_bytes)):
            stats[key] = max(stats[key], value)
        if local_callback:
            local_callback(dict(site=j, to_right=direction, iterations=iterations,
                                packed_input=layout.size, packed_output=out.size,
                                dense_output=math.prod(out.shape), blocks=action.block_count,
                                batches=action.batch_count, coefficient_bytes=action.coefficient_bytes,
                                setup_seconds=setup_seconds, seconds=time.perf_counter()-started,
                                max_bond=max(a.shape[-1] for a in state), storage="host"))
        del action, a, t, delta, candidate, le, re
        images[end if direction else j] = overlaps[end if direction else j] = None
        index = j if direction else end-1
        moving_image = step(moving_image, index, direction)
        if direction:
            moving_overlap = _overlap_left(moving_overlap, trace[index], xp.asarray(state[index]))
        else:
            moving_overlap = _overlap_right(moving_overlap, trace[index], xp.asarray(state[index]))
    return stats


def solve_host(liouvillian, initial, trace, cfg, paired, callback, local_callback, checkpoint_callback):
    from .ness import NessResult, _overlap_left
    started = time.perf_counter()
    guess = compress(initial if initial is not None else trace, cfg.bond_cap, cfg.cutoff)
    amplitude = inner(trace, guess)
    if abs(amplitude) < np.finfo(float).tiny:
        raise ValueError("Initial/compressed state has zero trace")
    guess = guess.scale(1/amplitude)
    guess.ensure_right_canonical()
    guess.move_qnidx(guess.site_num)
    state = [np.array(a.array, dtype=complex, copy=True) for a in guess]
    state[0] *= complex(guess.coeff)
    charges = [q.copy() for q in guess.qn]
    physical = [np.asarray(b.sigmaqn) for b in guess.model.basis]
    prefix_mpo = liouvillian.copy()
    prefix_mpo.move_qnidx(prefix_mpo.site_num)
    mpo_charges = [q.copy() for q in prefix_mpo.qn]
    del prefix_mpo
    operator = [xp.asarray(a.array, dtype=complex) for a in liouvillian]
    trace_cores = [xp.asarray(a.array, dtype=complex) for a in trace]
    trace_cores[0] = trace_cores[0]*complex(trace.coeff)
    identity = [xp.eye(len(q), dtype=complex)[None, :, :, None] for q in physical]
    zero_mpo = [np.zeros((1, physical[0].shape[1]), int) for _ in charges]
    allowed = sector_prefixes(physical, guess.qntot)
    prefix_trace = trace.copy()
    prefix_trace.move_qnidx(trace.site_num)
    trace_charges = [q.copy() for q in prefix_trace.qn]
    adjoint = [w.conj().swapaxes(1, 2) for w in operator]
    defect = (image_norm(adjoint, trace_cores, trace_charges, [-q for q in mpo_charges], physical, allowed)
              / image_norm(identity, trace_cores, trace_charges, zero_mpo, physical, allowed))
    del adjoint, prefix_trace
    if defect > cfg.validation_tolerance:
        raise ValueError(f"L† trace is nonzero: {defect:.3e}; check sector/vectorization")
    physical_maps = [PhysicalMap([liouvillian[k].array for k in range(j, min(j+2, len(state)))])
                     for j in range(max(1, len(state)-1))]
    dimensions = [math.isqrt(len(q)) for q in physical]
    if paired:
        for q, d in zip(physical, dimensions):
            transposed = q.reshape(d, d, -1).swapaxes(0, 1).reshape(q.shape)
            if not np.array_equal(transposed, q[:, ::-1]):
                raise ValueError("Host-backed Hermiticity audit requires paired/reversible charge labels")

    def audit(sweep, stats=None):
        norm = image_norm(identity, state, charges, zero_mpo, physical, allowed)
        absolute = image_norm(operator, state, charges, mpo_charges, physical, allowed)
        overlap = xp.ones((1, 1), complex)
        for t, a in zip(trace_cores, state):
            overlap = _overlap_left(overlap, t, xp.asarray(a))
        herm = None
        if paired:
            def difference():
                for j, (a, d) in enumerate(zip(state, dimensions)):
                    a = xp.asarray(a)
                    b = a.reshape(a.shape[0], d, d, a.shape[2]).swapaxes(1, 2).conj().reshape(a.shape)
                    if len(state) == 1:
                        yield a-b
                    elif j == 0:
                        yield xp.concatenate((a, -b), axis=2)
                    elif j == len(state)-1:
                        yield xp.concatenate((a, b), axis=0)
                    else:
                        c = xp.zeros((2*a.shape[0], a.shape[1], 2*a.shape[2]), complex)
                        c[:a.shape[0], :, :a.shape[2]], c[a.shape[0]:, :, a.shape[2]:] = a, b
                        yield c
            dq = [charges[0]]+[np.concatenate((q, q[:, ::-1])) for q in charges[1:-1]]+[charges[-1]]
            herm = image_norm(identity, difference(), dq, zero_mpo, physical, allowed)/norm
        record = dict(sweep=sweep, residual=absolute/norm, absolute_residual=absolute,
                      trace_error=abs(complex(overlap[0, 0])-1), hermiticity_error=herm,
                      norm=norm, max_bond=max(a.shape[-1] for a in state), seconds=time.perf_counter()-started)
        if stats:
            for key in stats[0]:
                record[key] = (sum(s[key] for s in stats) if key in ("local_iterations", "matvecs")
                               else max(s[key] for s in stats))
        if callback:
            callback(dict(record))
        return record

    def converged(record):
        return (record["residual"] <= cfg.tolerance and record["trace_error"] <= cfg.trace_tolerance
                and (record["hermiticity_error"] is None or record["hermiticity_error"] <= cfg.hermiticity_tolerance))

    def export(cores, labels):
        output = Mps.from_mp(guess.model, cores)
        output.qn, output.qntot = [q.copy() for q in labels], guess.qntot.copy()
        output.qnidx = len(state)
        output.move_qnidx(0)
        output.to_right, output.coeff = True, 1.
        return output

    history = [audit(0)]
    best = history[0]
    best_state, best_charges = [a.copy() for a in state], [q.copy() for q in charges]
    for sweep in range(1, cfg.max_sweeps+1):
        if converged(best):
            break
        stats = [half_sweep(state, charges, physical, operator, trace_cores, direction, cfg,
                            mpo_charges, allowed, physical_maps, local_callback) for direction in (True, False)]
        record = audit(sweep, stats)
        history.append(record)
        if record["residual"] < best["residual"] or converged(record):
            best = record
            best_state, best_charges = [a.copy() for a in state], [q.copy() for q in charges]
        if checkpoint_callback:
            checkpoint_callback(export(best_state, best_charges), dict(best))
    return NessResult(export(best_state, best_charges), converged(best),
                      "converged" if converged(best) else "sweep_budget", best["residual"],
                      best["absolute_residual"], best["trace_error"], best["hermiticity_error"],
                      len(history)-1, "gpu" if USE_GPU else "cpu", time.perf_counter()-started, history, defect)
