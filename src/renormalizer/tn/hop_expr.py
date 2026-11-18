import os
from functools import lru_cache
import opt_einsum as oe

from renormalizer.mps.backend import np, xp
from renormalizer.mps.matrix import asxp
from renormalizer.mps.oe_contract_wrap import oe_contract, oe_contract_expression
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron


def hop_expr0(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    # zero site, used in tdvp time evolution
    # assuming the first index connects child and the second index connects parent
    # #--------o---------#
    # child--coeff--parent
    enode = ttne.node_list[ttns.node_idx[snode]]

    args = []
    input_indices = []
    output_indices = []
    shape = []

    tensor = enode.parent.environ_children[enode.idx_as_child]
    shape.append(tensor.shape[0])
    args.append(tensor)
    indices = ttne.get_child_indices(enode.parent, enode.idx_as_child, ttns, ttno)
    output_indices.append(indices[0])
    input_indices.append(indices[2])
    args.append(indices)

    tensor = enode.environ_parent
    shape.append(tensor.shape[0])
    args.append(tensor)
    indices = ttne.get_parent_indices(enode, ttns, ttno)
    assert len(indices) == 3
    indices[0] = tuple(list(indices[0]) + ["hop0_conj"])
    indices[2] = tuple(list(indices[2]) + ["hop0"])
    output_indices.append(indices[0])
    input_indices.append(indices[2])
    args.append(indices)

    expr = _contract_expression(args, shape, input_indices, output_indices)

    return expr


def hop_expr1(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron, return_hdiag=False):
    # build one site effective hamiltonian operator as an opt_einsum expression
    enode = ttne.node_list[ttns.node_idx[snode]]
    onode = ttno.node_list[ttns.node_idx[snode]]

    args = []
    # enode children environments
    for i, echild_environchild in enumerate(enode.environ_children):
        args.append(echild_environchild)
        args.append(ttne.get_child_indices(enode, i, ttns, ttno))
    # parent environments
    args.append(enode.environ_parent)
    args.append(ttne.get_parent_indices(enode, ttns, ttno))
    # operator
    args.extend([onode.tensor, ttno.get_node_indices(onode)])

    # input and output
    input_indices = ttns.get_node_indices(snode, ttno=ttno)
    output_indices = ttns.get_node_indices(snode, conj=True)

    shape = snode.shape
    # cache the contraction path
    expr = _contract_expression(args, shape, input_indices, output_indices)
    if not return_hdiag:
        return expr
    else:
        hdiag = _get_hdiag(args, input_indices, output_indices, shape)
        return expr, hdiag


def hop_expr2(snode: TreeNodeTensor, ttns: TTNS, ttno: TTNO, ttne: TTNEnviron):
    # build two-site effective hamiltonian operator as an opt_einsum expression
    sparent = snode.parent
    enode = ttne.node_list[ttns.node_idx[snode]]
    eparent = ttne.node_list[ttns.node_idx[sparent]]
    onode = ttno.node_list[ttns.node_idx[snode]]
    oparent = ttno.node_list[ttns.node_idx[sparent]]

    args = []
    # enode children environments
    for i, echild_environchild in enumerate(enode.environ_children):
        args.append(echild_environchild)
        args.append(ttne.get_child_indices(enode, i, ttns, ttno))

    # eparent children environments
    for i, enode_environchild in enumerate(eparent.environ_children):
        if eparent.children[i] is enode:
            continue
        args.append(enode_environchild)
        args.append(ttne.get_child_indices(eparent, i, ttns, ttno))

    # eparent parent environments
    args.append(eparent.environ_parent)
    args.append(ttne.get_parent_indices(eparent, ttns, ttno))

    # operator
    args.extend([oparent.tensor, ttno.get_node_indices(oparent)])
    args.extend([onode.tensor, ttno.get_node_indices(onode)])

    # input and output
    input_indices = ttns.get_node_indices(snode, include_parent=True, ttno=ttno)
    output_indices = ttns.get_node_indices(snode, conj=True, include_parent=True)

    # shape
    shape = list(snode.shape[:-1])
    shape_parent = list(snode.parent.shape)
    del shape_parent[snode.parent.children.index(snode)]
    shape += shape_parent
    # cache the contraction path
    expr = _contract_expression(args, shape, input_indices, output_indices)
    hdiag = _get_hdiag(args, input_indices, output_indices, shape)
    return expr, hdiag


@lru_cache(maxsize=512)
def _plan_sliced_hop(indices, shapes, limit):
    """Find an exact one-index tiling without an expensive multiway fallback."""
    path, info = oe.contract_path(indices, *shapes, shapes=True, optimize='optimal')
    if info.largest_intermediate <= limit:
        return path, None, 0, int(info.largest_intermediate), float(info.opt_cost)
    inputs, output = indices.split('->')
    labels = inputs.split(',')
    sizes = {}
    for label, shape in zip(labels, shapes):
        sizes.update(zip(label, shape))
    candidates = []
    for index, dimension in sizes.items():
        chunk = dimension//2
        while chunk:
            tiled_shapes = tuple(tuple(chunk if c == index else n for c,n in zip(label,shape))
                                 for label,shape in zip(labels, shapes))
            tiled_path, tiled_info = oe.contract_path(indices, *tiled_shapes, shapes=True, optimize='optimal')
            if tiled_info.largest_intermediate <= limit:
                cost = float(tiled_info.opt_cost)*int(np.ceil(dimension/chunk))
                candidates.append((cost, index, chunk, tiled_path, int(tiled_info.largest_intermediate)))
                break
            chunk //= 2
    if not candidates:
        raise MemoryError('No single-index TTNS hop tiling fits RENO_TTN_HOP_MEMORY_MB; increase the budget')
    cost, index, chunk, path, largest = min(candidates, key=lambda item:item[0])
    return path, index, chunk, largest, cost


def _contract_expression(args, x_shape, x_indices, y_indices):
    # contract_expression in interleaved format
    args_fake = args.copy()
    args_fake.extend([np.empty(x_shape), x_indices])
    args_fake.append(y_indices)
    indices, tensors = oe.parser.convert_interleaved_input(args_fake)
    args = [asxp(t) for t in tensors[:-1]] + [x_shape]
    # Optional cap for large tree two-site contractions. Unconstrained FLOP
    # optimization can otherwise build an effective-operator intermediate
    # much larger than the input/output wavefunction (tens of GB at D=64).
    limit_mb = os.environ.get("RENO_TTN_HOP_MEMORY_MB")
    if limit_mb is not None:
        limit_mb = float(limit_mb)
        if not np.isfinite(limit_mb) or limit_mb <= 0:
            raise ValueError("RENO_TTN_HOP_MEMORY_MB must be positive and finite")
        # Allow complex vectors even when the reference/operator is real.
        itemsize = max(16, np.result_type(*[t.dtype for t in tensors]).itemsize)
        limit = max(int(np.prod(x_shape)), int(limit_mb*1024**2/itemsize))
        shapes = tuple(tuple(t.shape) for t in tensors)
        path, sliced_index, chunk, _, _ = _plan_sliced_hop(indices, shapes, limit)
        if sliced_index is not None:
            inputs, output = indices.split('->')
            labels = inputs.split(',')
            sizes = {c:n for label,shape in zip(labels,shapes) for c,n in zip(label,shape)}
            def sliced_expression(matrix):
                operands = args[:-1]+[matrix]
                result = xp.zeros(tuple(sizes[c] for c in output), dtype=np.result_type(matrix.dtype, *[t.dtype for t in operands]))
                for start in range(0, sizes[sliced_index], chunk):
                    stop = min(start+chunk, sizes[sliced_index])
                    tiles = [t[tuple(slice(start, stop) if c == sliced_index else slice(None) for c in label)]
                             for t,label in zip(operands, labels)]
                    # Do not retain all tiles' precontracted constants. At most
                    # one tile and its intermediates are live at a time.
                    value = oe_contract(indices, *tiles, optimize=path)
                    if sliced_index in output:
                        target = tuple(slice(start, stop) if c == sliced_index else slice(None) for c in output)
                        result[target] = value
                    else:
                        result += value
                return result
            return sliced_expression
    expr = oe_contract_expression(
        indices,
        *args,
        constants=list(range(len(tensors)))[:-1],
    )
    return expr


def _get_hdiag(args, input_indices, output_indices=None, shape=None):
    """Contract the diagonal by identifying corresponding bra/ket axes.

    Explicit output indices also cover spectator physical axes (e.g. a
    purification) and environments built from separately named states.
    """
    from renormalizer.tn.contractions import bounded_contract

    if output_indices is not None:
        if len(input_indices) != len(output_indices):
            raise ValueError("Hamiltonian input and output ranks must agree")
        replacements = dict(zip(output_indices, input_indices))
    else:
        # Backward-compatible convention for callers using the same TTNS
        # for bra and ket. Core callers pass the explicit output labels.
        replacements = {}
        for indices in args[1::2]:
            for index in indices:
                label = list(index)
                if isinstance(label[0], str) and label[0].endswith("_conj"):
                    label[0] = label[0][:-5]
                elif label[0] == "up":
                    label[0] = "down"
                replacements[index] = tuple(label)
    new_args = []
    for tensor, indices in zip(args[::2], args[1::2]):
        # Each operand has a LIST of index tuples. Identify bra with ket
        # within every tuple; changing the outer list instead sums over bra
        # indices and returns column sums, not the Hamiltonian diagonal.
        diagonal_indices = [replacements.get(index, index) for index in indices]
        new_args.extend([asxp(tensor), diagonal_indices])
    # A spectator axis is passed through unchanged by H and occurs only
    # in the dynamic state, not in any constant operand. Its diagonal is 1.
    present = {index for indices in new_args[1::2] for index in indices}
    for axis, index in enumerate(input_indices):
        if index not in present:
            if shape is None:
                raise ValueError("Local shape is required for spectator axes")
            new_args.extend([xp.ones(shape[axis]), [index]])
    return bounded_contract(*new_args, input_indices)
