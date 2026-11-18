"""GPU-resident symmetric one-site TTNS projector splitting.

Uses the same charged QR and bounded local exponentials as the audited PS2
implementation. Bond spaces are fixed (QR can remove impossible dimensions).
No SVD truncation or normalization is performed. Rank convergence must still
be tested: conservation of norm/energy alone does not prove accurate dynamics.
"""
from .gpu_tree import DeviceEnvironment, local_expression
from .bounded_linalg import expm_bounded
from renormalizer.tn.contractions import bounded_contract


def tdvp_one_site(state, operator, dt, *, krylov_tol=1e-11,
                  krylov_bytes=128*1024**2):
    """Mutate a root-canonical, complex device state by a second-order step.

    The second traversal reverses the child order as well as the elementary
    operations of the first traversal. Negative edge evolution subtracts the
    overlap between adjacent one-site tangent projectors.
    """
    env = DeviceEnvironment(state, operator)
    infos = []

    def evolve(value, action, step):
        result, info = expm_bounded(lambda x: action(x.reshape(value.shape)).ravel(),
            -1j*step, value.ravel(), rtol=krylov_tol, basis_bytes=krylov_bytes, max_order=320)
        infos.append(info)
        return result.reshape(value.shape)

    def site(node, step):
        node.tensor = evolve(node.tensor,
            local_expression(env, state.node_idx[node], pair=False), step)

    def edge(child, value, step):
        enode = env.node_list[state.node_idx[child]]
        up = enode.parent.environ_children[child.idx_as_child]
        down = enode.environ_parent
        # Coefficient axes are (child, parent); environments are (bra, op, ket).
        return evolve(value, lambda x: bounded_contract(
            up, ['a', 'o', 'b'], down, ['c', 'o', 'd'],
            x, ['b', 'd'], ['a', 'c']), step)

    def forward(node):
        for axis, child in enumerate(node.children):
            state.push_cano_to_child(node, axis)
            env.down(state.node_idx[node], axis)
            forward(child)
            coefficient = state.decompose_to_parent(child)
            env.up(state.node_idx[child])
            coefficient = edge(child, coefficient.T, -dt/2)
            state.merge_to_parent(child, coefficient.T)
        site(node, dt/2)

    def backward(node):
        site(node, dt/2)
        for axis in range(len(node.children)-1, -1, -1):
            child = node.children[axis]
            coefficient = state.decompose_to_child(node, axis)
            env.down(state.node_idx[node], axis)
            coefficient = edge(child, coefficient, -dt/2)
            state.merge_to_child(node, axis, coefficient)
            backward(child)
            state.push_cano_to_parent(child)
            env.up(state.node_idx[child])

    forward(state.root)
    backward(state.root)
    return dict(local_exponentials=len(infos),
        local_h_actions=sum(i['h_actions'] for i in infos),
        local_krylov_vectors=sum(i['vectors'] for i in infos),
        two_pass_exponentials=sum(i['two_pass'] for i in infos))
