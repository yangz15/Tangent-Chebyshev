"""Resident, zero-charge TTNS kernels using Renormalizer's tree conventions.

Only scalar convergence decisions leave the GPU inside sweeps.  Renormalizer
provides symbolic TTNOs, index conventions, and the frozen tangent
implementation. CuPy supplies the ground-state Lanczos solver. This module does not
monkey-patch the shared Renormalizer checkout.
"""
from functools import lru_cache
import copy
import math
import os
import time
import numpy as np
import cupy as cp
import opt_einsum as oe

from renormalizer.tn import TTNS, TTNO
from renormalizer.tn.node import TreeNodeTensor, copy_connection
from renormalizer.tn.tree import TTNEnviron
from renormalizer.tn.hop_expr import _get_hdiag
from renormalizer.lib import expm_krylov
from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
from .bounded_linalg import expm_bounded, partial_svd
from renormalizer.tn.contractions import bounded_contract, bounded_local_expression

_add_scaled = cp.ElementwiseKernel('T value, T coefficient', 'T result',
                                  'result += coefficient * value;', 'xy_factorized_add')


@lru_cache(maxsize=8192)
def contraction_path(equation, shapes):
    return oe.contract_path(equation, *shapes, shapes=True, optimize="auto-hq")[0]


@lru_cache(maxsize=4096)
def bounded_path(equation, shapes, elements):
    for factor in (1, 2, 4, 8):
        try:
            return oe.contract_path(equation, *shapes, shapes=True, optimize='auto-hq',
                                    memory_limit=elements*factor)[0]
        except RuntimeError:
            pass
    raise RuntimeError(f'No bounded contraction for {equation}, shapes={shapes}')


def contract(*args):
    return bounded_contract(*args)


class DeviceNode(TreeNodeTensor):
    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = cp.asarray(value)

    def check_canonical(self, atol=1e-8, assertion=True):
        a = self.tensor.reshape(-1, self.shape[-1])
        ok = bool(cp.allclose(a.conj().T @ a, cp.eye(a.shape[1]), atol=atol))
        if assertion:
            assert ok
        return ok


def device_operator(operator):
    if os.environ.get('RENO_HOST_OPERATORS')=='1':
        return operator
    nodes = [DeviceNode(n.tensor, n.qn) for n in operator]
    return TTNO(operator.basis, operator.terms,
                root=copy_connection(operator.node_list, nodes))


class DeviceState(TTNS):
    @classmethod
    def from_host(cls, state, chi, svd_cutoff=0.):
        nodes = [DeviceNode(n.tensor, n.qn) for n in state]
        result = cls(state.basis, root=copy_connection(state.node_list, nodes))
        result.chi = chi
        result.svd_cutoff = svd_cutoff
        result.discarded = []
        result.svd_records = []
        result.svd_seconds = 0.
        result.bond_caps = None
        return result

    def copy(self):
        nodes = [DeviceNode(n.tensor.copy(), n.qn.copy()) for n in self]
        result = type(self)(self.basis, root=copy_connection(self.node_list, nodes))
        result.chi = self.chi
        result.svd_cutoff = self.svd_cutoff
        result.discarded = []
        result.svd_records = []
        result.svd_seconds = 0.
        result.bond_caps = None if self.bond_caps is None else self.bond_caps.copy()
        return result

    def host(self):
        nodes = [TreeNodeTensor(cp.asnumpy(n.tensor), n.qn) for n in self]
        return TTNS(self.basis, root=copy_connection(self.node_list, nodes))

    def to_complex(self):
        new = self.copy()
        for node in new:
            node.tensor = node.tensor.astype(cp.complex128)
        return new

    def scale(self, value, inplace=False):
        result = self if inplace else self.copy()
        result.root.tensor *= value
        return result

    def decompose_to_parent(self, node):
        q, r = cp.linalg.qr(node.tensor.reshape(-1, node.shape[-1]), mode="reduced")
        node.tensor = q.reshape(*node.shape[:-1], q.shape[1])
        node.qn = np.zeros((q.shape[1], self.basis.qn_size), dtype=int)
        # Renormalizer uses (old-parent, new-parent), without conjugation.
        return r.T

    def merge_to_parent(self, node, value):
        parent = node.parent
        axis = node.idx_as_child
        parent.tensor = cp.moveaxis(cp.tensordot(parent.tensor, value, axes=(axis, 0)), -1, axis)

    def decompose_to_child(self, node, axis):
        moved = cp.moveaxis(node.tensor, axis, -1)
        q, r = cp.linalg.qr(moved.reshape(-1, moved.shape[-1]), mode="reduced")
        node.tensor = cp.moveaxis(q.reshape(*moved.shape[:-1], q.shape[1]), -1, axis)
        node.children[axis].qn = np.zeros((q.shape[1], self.basis.qn_size), dtype=int)
        return r.T

    def merge_to_child(self, node, axis, value):
        child = node.children[axis]
        child.tensor = cp.tensordot(child.tensor, value, axes=(-1, 0))

    def canonicalise(self):
        for node in self.postorder_list():
            if node.parent is not None:
                self.push_cano_to_parent(node)
        return self

    def merge_with_parent(self, node):
        return contract(node.tensor, self.get_node_indices(node),
                        node.parent.tensor, self.get_node_indices(node.parent),
                        self.get_node_indices(node, include_parent=True))

    def update_2site(self, node, tensor, cano_parent=True):
        cp.cuda.get_current_stream().synchronize()
        begun = time.perf_counter()
        left_shape = node.shape[:-1]
        parent = node.parent
        axis = node.idx_as_child
        right_shape = parent.shape[:axis] + parent.shape[axis+1:]
        matrix = tensor.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
        cap = self.chi if self.bond_caps is None else min(self.chi,self.bond_caps[self.node_idx[node]])
        use_partial = min(matrix.shape) > max(512, 4*cap)
        if use_partial:
            u, s, vh, info = partial_svd(matrix, cap)
        else:
            u, s, vh = cp.linalg.svd(matrix, full_matrices=False)
            info = dict(solver='full_svd', relative_singular_residual=0.)
        self.svd_records.append(info)
        rank = min(cap, len(s))
        if self.svd_cutoff > 0 and not use_partial:
            tail = cp.cumsum((s*s)[::-1])
            removable = int(cp.searchsorted(tail, self.svd_cutoff*tail[-1], side="right"))
            rank = max(1, min(rank, len(s)-removable))
        # Record actual discarded norm, not a proxy based on nominal bond cap.
        self.discarded.append(cp.maximum(0., cp.vdot(matrix, matrix).real-cp.sum(s[:rank]**2))
                              if use_partial else cp.sum(s[rank:] ** 2))
        u, sr, vh = u[:, :rank], s[:rank], vh[:rank]
        if cano_parent:
            vh = sr[:, None] * vh
        else:
            u = u * sr[None, :]
        node.tensor = u.reshape(*left_shape, rank)
        parent.tensor = cp.moveaxis(vh.reshape(rank, *right_shape), 0, axis)
        node.qn = np.zeros((rank, self.basis.qn_size), dtype=int)
        cp.cuda.get_current_stream().synchronize()
        self.svd_seconds += time.perf_counter()-begun

    def set_bond_profile(self, profile, length):
        if profile == 'uniform':
            self.bond_caps = None
            return
        if profile != 'interval':
            raise ValueError(profile)
        self.bond_caps = {}
        sites = {}
        for node in self.postorder_list():
            physical = {dof for basis in self.tn2bn[node].basis_sets for dof in basis.dofs
                        if isinstance(dof,(int,np.integer))}
            physical.update(site for child in node.children for site in sites[child])
            sites[node] = physical
            if node.parent is None:
                continue
            count = len(physical)
            cap = self.chi
            if 0 in physical or length-1 in physical:
                cap = min(cap,64)
            elif count <= 8:
                cap = min(cap,96)
            elif count <= 16:
                cap = min(cap,128)
            self.bond_caps[self.node_idx[node]] = min(cap,2**min(count,20))
        # Interval lengths alone do not bound a two-node center: four
        # exterior bonds can meet at an internal edge. Enforce the bound
        # on those actual products, reducing only caps that violate it.
        center_limit = 4 * 1024**3
        for node in self.node_list[1:]:
            parent = node.parent
            exterior = list(node.children)
            exterior += [child for child in parent.children if child is not node]
            if parent is not self.root:
                exterior.append(parent)
            physical_size = int(np.prod(node.shape[len(node.children):-1]))
            physical_size *= int(np.prod(parent.shape[len(parent.children):-1]))
            def center_bytes():
                return 16 * physical_size * math.prod(
                    self.bond_caps[self.node_idx[edge]] for edge in exterior)
            while center_bytes() > center_limit:
                edge = max(exterior, key=lambda item: self.bond_caps[self.node_idx[item]])
                index = self.node_idx[edge]
                cap = self.bond_caps[index]
                allowed = max(1, cap * center_limit // center_bytes())
                self.bond_caps[index] = max(1, 2 * (allowed // 2))

    @property
    def norm(self):
        # Caller must have the center at root.
        return float(cp.linalg.norm(self.root.tensor))

    def discarded_summary(self):
        if not self.discarded:
            return dict(discarded_sum=0., discarded_max=0., svd_count=0)
        values = cp.stack(self.discarded)
        return dict(discarded_sum=float(values.sum()), discarded_max=float(values.max()),
                    svd_count=len(self.discarded))


class DeviceEnvironment(TTNEnviron):
    """Directed environments. Initialize only upward messages for a root sweep."""
    def __init__(self, ket, operator, bra=None):
        self.ket, self.operator, self.bra = ket, operator, ket if bra is None else bra
        super().__init__(ket, operator, build_environ=False, bra=self.bra)
        self.root.environ_parent = cp.ones((1, 1, 1))
        self.build_children_environ(ket, operator, bra=self.bra)

    def _node_args(self, index, omit_child=None, omit_parent=False):
        ket, op, bra = self.ket, self.operator, self.bra
        e, s, o, b = (x.node_list[index] for x in (self, ket, op, bra))
        args = []
        for j, message in enumerate(e.environ_children):
            if j != omit_child:
                args += [message, self.get_child_indices(e, j, ket, op, bra=bra)]
        if not omit_parent:
            args += [e.environ_parent, self.get_parent_indices(e, ket, op, bra=bra)]
        args += [b.tensor.conj(), bra.get_node_indices(b, conj=True),
                 o.tensor, op.get_node_indices(o),
                 s.tensor, ket.get_node_indices(s, ttno=op)]
        return args

    def up(self, index):
        s = self.ket.node_list[index]
        if s.parent is None:
            return
        e = self.node_list[index]
        args = self._node_args(index, omit_parent=True)
        args += [self.get_parent_indices(e, self.ket, self.operator, bra=self.bra)]
        if not e.parent.environ_children:
            e.parent.environ_children = [None] * len(e.parent.children)
        e.parent.environ_children[e.idx_as_child] = contract(*args)

    def down(self, index, axis):
        e = self.node_list[index]
        args = self._node_args(index, omit_child=axis)
        args += [self.get_child_indices(e, axis, self.ket, self.operator, bra=self.bra)]
        e.children[axis].environ_parent = contract(*args)

    def build_children_environ_node(self, snode, ttns, ttno, bra=None):
        self.up(ttns.node_idx[snode])

    def build_parent_environ_node(self, snode, axis, ttns, ttno, bra=None):
        self.down(ttns.node_idx[snode], axis)

    def scalar(self):
        args = self._node_args(0) + [[]]
        return contract(*args).reshape(())


def factorized_dummy_action(environments, operators, inputs, outputs, shape):
    """Sparse TTNO-core sum of mode products for large dummy-node centers.

    This avoids dense effective-operator products and virtual-rank enlarged
    intermediates. Batched GEMMs apply one external matrix along one tensor
    axis, with contiguous input/output and no full-tensor transpose copy.
    """
    ket_axes = {label:axis for axis,label in enumerate(inputs)}
    bra_axes = {label:axis for axis,label in enumerate(outputs)}
    covered = {labels[2] for _,labels in environments}
    if any(shape[axis] != 1 for axis,label in enumerate(inputs) if label not in covered):
        return None
    if any(ket_axes[labels[2]] != bra_axes[labels[0]] for _,labels in environments):
        return None
    core_args = []
    for tensor,labels in operators:
        core_args += [tensor, labels]
    core = cp.asnumpy(contract(*core_args, [labels[1] for _,labels in environments]))
    matrices = {}
    terms = []
    for coordinates in zip(*np.nonzero(core)):
        coefficient = core[coordinates].item()
        factors = []
        for edge,(tensor,labels) in enumerate(environments):
            key = (edge,coordinates[edge])
            if key not in matrices:
                matrix = tensor[:,coordinates[edge],:]
                scalar = cp.trace(matrix)/matrix.shape[0]
                deviation = float(cp.linalg.norm(matrix-scalar*cp.eye(matrix.shape[0])))
                is_identity = deviation <= 1e-12*max(1.,float(cp.linalg.norm(matrix)))
                matrices[key] = (matrix,scalar.item() if is_identity else None)
            matrix,scalar = matrices[key]
            if scalar is None:
                factors.append((ket_axes[labels[2]],matrix))
            else:
                coefficient *= scalar
        if coefficient != 0:
            terms.append((coefficient,factors))
    def action(value):
        value = value.reshape(shape)
        result = cp.zeros_like(value)
        for coefficient,factors in terms:
            work = value
            for axis,matrix in factors:
                prefix = int(np.prod(shape[:axis]))
                suffix = int(np.prod(shape[axis+1:]))
                if suffix == 1:
                    work = (work.reshape(prefix,shape[axis]) @ matrix.T).reshape(shape)
                else:
                    work = cp.matmul(matrix[None,:,:],
                                     work.reshape(prefix,shape[axis],suffix)).reshape(shape)
            _add_scaled(work, value.dtype.type(coefficient), result)
        return result
    action.factorized_terms = len(terms)
    return action


def local_expression(env, index, pair=False, diagonal=False):
    ket, op, bra = env.ket, env.operator, env.bra
    s = ket.node_list[index]
    selected = [index]
    if pair:
        selected.append(ket.node_idx[s.parent])
    selected_set = set(selected)
    args, environments, operators = [], [], []
    for i in selected:
        e, o = env.node_list[i], op.node_list[i]
        for j, child in enumerate(ket.node_list[i].children):
            if ket.node_idx[child] not in selected_set:
                labels = env.get_child_indices(e, j, ket, op, bra=bra)
                args += [e.environ_children[j], labels]
                environments.append((e.environ_children[j],labels))
        parent = ket.node_list[i].parent
        if parent is None or ket.node_idx[parent] not in selected_set:
            labels = env.get_parent_indices(e, ket, op, bra=bra)
            args += [e.environ_parent, labels]
            environments.append((e.environ_parent,labels))
        labels = op.get_node_indices(o)
        args += [o.tensor, labels]
        operators.append((o.tensor,labels))
    inputs = ket.get_node_indices(s, include_parent=pair, ttno=op)
    outputs = bra.get_node_indices(bra.node_list[index], include_parent=pair, conj=True)
    shape = s.shape
    if pair:
        p, axis = s.parent, s.idx_as_child
        shape = s.shape[:-1] + p.shape[:axis] + p.shape[axis+1:]
    if np.prod(shape) >= 4*1024**2 and not diagonal:
        factorized = factorized_dummy_action(environments,operators,inputs,outputs,shape)
        if factorized is not None:
            return factorized
    # Parse with a shape-only NumPy placeholder (no physical Hilbert-space array).
    fake = np.lib.stride_tricks.as_strided(np.empty(1), shape=shape, strides=(0,) * len(shape))
    equation, arrays = oe.parser.convert_interleaved_input(args + [fake, inputs, outputs])
    # Limit cached constant intermediates as well as each Krylov matvec.
    # An unconstrained FLOP-optimal path can retain multi-GB effective tensors.
    action = bounded_local_expression(equation, [cp.asarray(a) for a in arrays[:-1]], shape)
    if diagonal:
        return action, _get_hdiag(args, inputs, outputs, shape)
    return action


def matrix_element(bra, op, ket):
    return DeviceEnvironment(ket, op, bra=bra).scalar()


def ground_sweep(state, operator, tolerance=1e-8, acceptance=None):
    env = DeviceEnvironment(state, operator)
    energies = []
    def solve(child, toward_parent):
        i = state.node_idx[child]
        guess = state.merge_with_parent(child)
        action = local_expression(env, i, pair=True)
        calls = 0
        def hop(x):
            nonlocal calls
            calls += 1
            return action(x).ravel()
        size = guess.size
        if size <= 4:
            local = cp.stack([hop(column) for column in cp.eye(size, dtype=guess.dtype)], axis=1)
            eigenvalues, vectors = cp.linalg.eigh((local+local.conj().T)*.5)
        else:
            linear = LinearOperator((size, size), matvec=hop, dtype=guess.dtype)
            eigenvalues, vectors = eigsh(linear, k=1, which='SA', v0=guess.ravel(),
                ncv=min(max(4, min(40, (1024**3)//guess.nbytes)), size-1),
                maxiter=2400, tol=tolerance)
        value = vectors[:, 0]
        image = hop(value)
        energy = float(cp.vdot(value, image).real)
        residual = float(cp.linalg.norm(image-energy*value))
        limit = max(tolerance*10, 1e-7) if acceptance is None else acceptance
        if not np.isfinite(residual) or residual > limit:
            raise RuntimeError(f'GPU local Lanczos residual {residual:.3e} exceeds {limit:.3e}')
        if not hasattr(state, 'ground_solver_records'):
            state.ground_solver_records = []
        state.ground_solver_records.append(dict(actions=calls, residual=residual, target=tolerance))
        state.update_2site(child, value, cano_parent=toward_parent)
        energies.append(energy)
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
    return energies[-1]


def tdvp_step(state, operator, dt, krylov_tol=1e-8, krylov_bytes=256*1024**2):
    """Symmetric two-site projector splitting with only required messages.

    This is the Renormalizer tree PS2 traversal; no diagonal is constructed
    for exponential actions. Each message is updated only when its branch
    changes and a later local action consumes it.
    """
    env = DeviceEnvironment(state, operator)
    counts, h_actions, two_pass = [], [], []
    def evolve(node, pair, step):
        i = state.node_idx[node]
        value = state.merge_with_parent(node) if pair else node.tensor
        action = local_expression(env, i, pair=pair)
        result, info = expm_bounded(lambda x: action(x).ravel(), -1j*step,
                                   value.ravel(), rtol=krylov_tol, basis_bytes=krylov_bytes)
        counts.append(info['vectors'])
        h_actions.append(info['h_actions'])
        two_pass.append(info['two_pass'])
        return result.reshape(value.shape)
    def forward(node):
        for axis, child in enumerate(node.children):
            if child.children:
                state.push_cano_to_child(node, axis)
                env.down(state.node_idx[node], axis)
                forward(child)
            value = evolve(child, True, dt/2)
            state.update_2site(child, value, cano_parent=True)
            env.up(state.node_idx[child])
            if not (node is state.root and axis == len(node.children)-1):
                node.tensor = evolve(node, False, -dt/2)
    def backward(node):
        for axis, child in reversed(list(enumerate(node.children))):
            if not (node is state.root and axis == len(node.children)-1):
                node.tensor = evolve(node, False, -dt/2)
            value = evolve(child, True, dt/2)
            state.update_2site(child, value, cano_parent=not child.children)
            if child.children:
                env.down(state.node_idx[node], axis)
                backward(child)
                state.push_cano_to_parent(child)
            env.up(state.node_idx[child])
    forward(state.root)
    backward(state.root)
    return dict(local_krylov_vectors=sum(counts), local_exponentials=len(counts),
                local_h_actions=sum(h_actions), two_pass_exponentials=sum(two_pass))


def variational_sum(guess, terms, sweeps=2):
    """Fit sum c_a O_a |ket_a> directly, without expanded TTNS intermediates.

    Two-site ALS followed by one bond SVD enables rank growth and controlled
    truncation. Mixed environments are reused during each sweep.
    """
    result = guess.copy()
    for _ in range(sweeps):
        envs = [DeviceEnvironment(ket, op, bra=result) for _, op, ket in terms]
        def fit(child, toward_parent):
            i = result.node_idx[child]
            value = None
            for (coefficient, _, ket), env in zip(terms, envs):
                term = coefficient * local_expression(env, i, pair=True)(
                    ket.merge_with_parent(ket.node_list[i]))
                value = term if value is None else value + term
            result.update_2site(child, value, cano_parent=toward_parent)
        def visit(node):
            for axis, child in enumerate(node.children):
                if child.children:
                    fit(child, False)
                    for env in envs:
                        env.down(result.node_idx[node], axis)
                    visit(child)
                fit(child, True)
                for env in envs:
                    env.up(result.node_idx[child])
        visit(result.root)
    return result
