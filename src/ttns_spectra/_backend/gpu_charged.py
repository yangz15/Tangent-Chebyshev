"""Charge-preserving GPU TTNS sweeps, using the validated tree PS2 kernels.

Only small quantum-number labels and scalar decisions are copied to the CPU.
Tensor contractions, block QR/SVD, and local Krylov vectors stay on the GPU.
"""
import numpy as np
import cupy as cp
import time
from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
from renormalizer.mps.svd_qn import add_outer
from .gpu_tree import (
    DeviceState, DeviceEnvironment, local_expression, device_operator,
    matrix_element, tdvp_step, variational_sum)


def block_factor(matrix, qleft, qright, total, cap=None):
    """Reduced block QR (cap=None) or globally truncated block SVD."""
    qleft, qright = qleft.reshape(-1, len(total)), qright.reshape(-1, len(total))
    blocks, singular, labels = [], [], []
    offset = 0
    for charge in np.unique(qleft, axis=0):
        rows = np.flatnonzero(np.all(qleft == charge, axis=1))
        cols = np.flatnonzero(np.all(qright == total-charge, axis=1))
        if not len(cols):
            continue
        rows_d, cols_d = cp.asarray(rows), cp.asarray(cols)
        block = matrix[rows_d[:, None], cols_d[None, :]]
        if cap is None:
            u, v = cp.linalg.qr(block, mode='reduced')
            s = cp.ones(u.shape[1])
        else:
            u, s, v = cp.linalg.svd(block, full_matrices=False)
        blocks.append((rows_d, cols_d, u, v, offset, len(s)))
        singular.append(s)
        labels.extend([charge]*len(s))
        offset += len(s)
    if not singular:
        raise ValueError('empty quantum-number decomposition')
    singular = cp.concatenate(singular)
    selected = (np.arange(offset) if cap is None else
                cp.asnumpy(cp.argsort(singular)[::-1][:min(cap, offset)]))
    ufull = cp.zeros((matrix.shape[0], len(selected)), dtype=matrix.dtype)
    vfull = cp.zeros((len(selected), matrix.shape[1]), dtype=matrix.dtype)
    for rows, cols, u, v, offset, size in blocks:
        positions = np.flatnonzero((offset <= selected) & (selected < offset+size))
        if not len(positions):
            continue
        local, positions_d = cp.asarray(selected[positions]-offset), cp.asarray(positions)
        ufull[rows[:, None], positions_d[None, :]] = u[:, local]
        vfull[positions_d[:, None], cols[None, :]] = v[local]
    retained = singular[cp.asarray(selected)]
    discarded = cp.maximum(0., cp.sum(singular**2)-cp.sum(retained**2)) if cap is not None else cp.asarray(0.)
    return ufull, retained, vfull, np.asarray(labels)[selected], discarded


class ChargedState(DeviceState):
    def decompose_to_parent(self, node):
        ql, qr, _ = self.get_qnmat(node)
        u, _, v, qnew, _ = block_factor(node.tensor.reshape(-1, node.shape[-1]), ql, qr, self.qntot)
        node.tensor = u.reshape(*node.shape[:-1], u.shape[1])
        node.qn = qnew
        return v.T

    def decompose_to_child(self, node, axis):
        ql = np.zeros(self.basis.qn_size, dtype=int)
        for i, child in enumerate(node.children):
            if i != axis:
                ql = add_outer(ql, child.qn)
        for basis in self.tn2bn[node].basis_sets:
            ql = add_outer(ql, basis.sigmaqn)
        ql = add_outer(ql, self.qntot-node.qn)
        moved = cp.moveaxis(node.tensor, axis, -1)
        u, _, v, qnew, _ = block_factor(moved.reshape(-1, moved.shape[-1]),
                                        ql, node.children[axis].qn, self.qntot)
        node.tensor = cp.moveaxis(u.reshape(*moved.shape[:-1], u.shape[1]), -1, axis)
        node.children[axis].qn = self.qntot-qnew
        return v.T

    def update_2site(self, node, tensor, cano_parent=True):
        parent, axis = node.parent, node.idx_as_child
        left_shape = node.shape[:-1]
        right_shape = parent.shape[:axis]+parent.shape[axis+1:]
        ql, qr, _ = self.get_qnmat(node, include_parent=True)
        u, s, v, qnew, loss = block_factor(tensor.reshape(int(np.prod(left_shape)), -1),
                                          ql, qr, self.qntot, cap=self.chi)
        self.discarded.append(loss)
        if cano_parent:
            v = s[:, None]*v
        else:
            u = u*s[None, :]
        node.tensor = u.reshape(*left_shape, len(s))
        node.qn = qnew
        parent.tensor = cp.moveaxis(v.reshape(len(s), *right_shape), 0, axis)


def ground_sweep(state, operator, tolerance=1e-9, *, solver='lanczos', callback=None):
    """Charge-preserving two-site sweep, with independently verified residuals.

    ``davidson`` uses the exact local diagonal as its preconditioner. Solver
    selection does not change the variational space or truncation. Optional
    callback receives one record after each local optimization.
    """
    if solver not in ('lanczos', 'davidson'):
        raise ValueError('ground solver must be lanczos or davidson')
    from renormalizer.tn.gs import davidson_gpu_lowest
    env = DeviceEnvironment(state, operator)
    records = []
    def solve(child, toward_parent):
        cp.cuda.get_current_stream().synchronize()
        begun = time.perf_counter()
        value = state.merge_with_parent(child)
        mask = cp.asarray(state.get_qnmask(child, include_parent=True))
        shape = value.shape
        guess = value[mask]
        if solver == 'davidson':
            action, diagonal = local_expression(env, state.node_idx[child], pair=True, diagonal=True)
            diagonal = diagonal[mask]
        else:
            action = local_expression(env, state.node_idx[child], pair=True)
        cp.cuda.get_current_stream().synchronize()
        setup_seconds = time.perf_counter()-begun
        actions = 0
        def hop(vector):
            nonlocal actions
            actions += 1
            tensor = cp.zeros(shape, dtype=vector.dtype)
            tensor[mask] = vector
            return action(tensor)[mask]
        size = len(guess)
        if size <= 3:
            h = cp.stack([hop(col) for col in cp.eye(size, dtype=guess.dtype)], axis=1)
            _, vectors = cp.linalg.eigh((h+h.conj().T)*.5)
            v = vectors[:, 0]
        elif solver == 'davidson':
            _, v = davidson_gpu_lowest(hop, diagonal, guess, tol=1e-12,
                residual_tol=tolerance, max_cycle=240, max_space=24)
        else:
            op = LinearOperator((size, size), matvec=hop, dtype=guess.dtype)
            # Bound the packed Lanczos basis for D64/D128 large local centers.
            # Keep the independent residual acceptance test unchanged.
            ncv_cap=min(size-1,max(0,(8*1024**3)//max(1,guess.nbytes)))
            if ncv_cap<3:raise MemoryError('Packed extremal Krylov basis exceeds 8 GiB budget')
            _, vectors = eigsh(op, k=1, which='SA', v0=guess,
                              ncv=min(ncv_cap,32), maxiter=3000, tol=tolerance)
            v = vectors[:, 0]
        hv = hop(v)
        energy = float(cp.vdot(v, hv).real)
        residual = float(cp.linalg.norm(hv-energy*v))
        # Restart an unconverged local Lanczos solve with more Krylov vectors.
        # Keep the independent acceptance threshold unchanged.
        if solver == 'lanczos' and size > 3:
            for retry_ncv in (64, 128):
                if residual <= max(1e-7, 30*tolerance):
                    break
                print(f'Retrying local eigensolve: size={size}, residual={residual}, ncv={min(size-1,retry_ncv)}', flush=True)
                _, vectors = eigsh(op, k=1, which='SA', v0=v,
                    ncv=min(ncv_cap, retry_ncv), maxiter=10000, tol=min(tolerance, 1e-11))
                v = vectors[:, 0]
                hv = hop(v)
                energy = float(cp.vdot(v, hv).real)
                residual = float(cp.linalg.norm(hv-energy*v))
        if residual > max(1e-7, 30*tolerance):
            raise RuntimeError(f'local eigensolver residual {residual}')
        tensor = cp.zeros(shape, dtype=v.dtype)
        tensor[mask] = v
        state.update_2site(child, tensor, cano_parent=toward_parent)
        records.append(residual)
        cp.cuda.get_current_stream().synchronize()
        if callback is not None:
            callback(dict(node=state.node_idx[child],toward_parent=toward_parent,
                solver=solver,dimension=size,full_dimension=value.size,h_actions=actions,
                residual=residual,energy=energy,setup_seconds=setup_seconds,
                total_seconds=time.perf_counter()-begun))
    def visit(node):
        for axis, child in enumerate(node.children):
            if child.children:
                solve(child, False)
                env.down(state.node_idx[node], axis)
                visit(child)
            solve(child, True)
            env.up(state.node_idx[child])
    visit(state.root)
    state.root.tensor /= cp.linalg.norm(state.root.tensor)
    return max(records)
