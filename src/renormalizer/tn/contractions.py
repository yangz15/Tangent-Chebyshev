"""Exact tree contractions with an explicit intermediate-memory budget.

Bounded opt_einsum paths can fall back to a many-input einsum, which is not
an efficient GPU plan. Require pairwise steps; if necessary tile output or
summed indices until a pairwise path fits. Tiling changes summation order,
never tensor ranks, matrix elements or precision. Inputs, final output and
BLAS workspace are additional to the per-intermediate budget.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
import math
import os
import numpy as np
import opt_einsum as oe
from renormalizer.mps.backend import xp, USE_GPU

WORKSPACE_BYTES = int(os.environ.get('RENO_TTN_WORKSPACE_MB', '256'))*2**20
if WORKSPACE_BYTES <= 0:
    raise ValueError('RENO_TTN_WORKSPACE_MB must be positive')


@dataclass
class ContractionPlan:
    equation: str
    shapes: tuple
    path: list
    chunks: dict
    largest_intermediate: int
    expression: object
    # Cache programs, never operands or intermediate arrays. Boundary tiles
    # can have different shapes, but the same binary contraction path works.
    _tile_expressions: dict = field(default_factory=dict, init=False, repr=False)

    def __call__(self, *arrays, backend=None):
        backend = backend or ('cupy' if USE_GPU else 'numpy')
        if not self.chunks:
            return self.expression(*arrays, backend=backend)
        inputs, output = self.equation.split('->')
        inputs = inputs.split(',')
        dimensions = {label: extent for labels, shape in zip(inputs, self.shapes)
                      for label, extent in zip(labels, shape)}
        module = xp if backend == 'cupy' else np
        result = module.zeros(tuple(dimensions[i] for i in output),
                              dtype=module.result_type(*arrays))
        labels = tuple(self.chunks)
        ranges = [range(0, dimensions[i], self.chunks[i]) for i in labels]
        for starts in product(*ranges):
            slices = {label: slice(start, min(start+self.chunks[label], dimensions[label]))
                      for label, start in zip(labels, starts)}
            operands = [a[tuple(slices.get(i, slice(None)) for i in indices)]
                        for a, indices in zip(arrays, inputs)]
            # The final tile may be smaller. The same binary path remains
            # valid and its intermediate sizes can only decrease.
            shapes = tuple(a.shape for a in operands)
            expression = self._tile_expressions.get(shapes)
            if expression is None:
                expression = oe.contract_expression(
                    self.equation, *shapes, optimize=self.path)
                self._tile_expressions[shapes] = expression
            value = expression(*operands, backend=backend)
            result[tuple(slices.get(i, slice(None)) for i in output)] += value
        return result


@lru_cache(maxsize=8192)
def contraction_plan(equation, shapes, itemsize=16, workspace_bytes=WORKSPACE_BYTES):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    dimensions = {}
    for labels, shape in zip(inputs, shapes):
        for label, size in zip(labels, shape):
            if label in dimensions and dimensions[label] != size:
                raise ValueError('broadcast indices are not supported by tree contraction tiling')
            dimensions[label] = size
    limit = max(1, workspace_bytes//itemsize)
    chunks = dict(dimensions)

    def plan_for(sizes, algorithm='auto-hq', bounded=True):
        dims = tuple(tuple(sizes[i] for i in labels) for labels in inputs)
        path, info = oe.contract_path(equation, *dims, shapes=True, optimize=algorithm,
                                      memory_limit=limit if bounded else None)
        return path, info

    for _ in range(128):
        try:
            path, info = plan_for(chunks)
        except RuntimeError:
            path, info = plan_for(chunks, bounded=False)
        if max(map(len, path), default=0) <= 2 and int(info.largest_intermediate) <= limit:
            sliced = {i:n for i,n in chunks.items() if n < dimensions[i]}
            expression = None if sliced else oe.contract_expression(equation, *shapes, optimize=path)
            return ContractionPlan(equation, shapes, path, sliced,
                                   int(info.largest_intermediate), expression)
        # Find a split that most reduces the largest pairwise intermediate.
        # Penalize total sliced FLOPs only after the memory score, to make
        # progress even if no one split can reach the target immediately.
        candidates = []
        for label, extent in chunks.items():
            if extent <= 1:
                continue
            trial = dict(chunks)
            trial[label] = (extent+1)//2
            _, trial_info = plan_for(trial, algorithm='greedy', bounded=False)
            tiles = math.prod(math.ceil(dimensions[i]/n) for i,n in trial.items())
            candidates.append((int(trial_info.largest_intermediate),
                               float(trial_info.opt_cost)*tiles, label, trial[label]))
        if not candidates:
            raise MemoryError('cannot construct a pairwise tree contraction within the memory budget')
        _, _, label, extent = min(candidates)
        chunks[label] = extent
    raise MemoryError('tree contraction exceeded the tiling-planner iteration limit')


def bounded_contract(*arguments):
    equation, arrays = oe.parser.convert_interleaved_input(arguments)
    arrays = [xp.asarray(a) for a in arrays]
    # Always reserve complex128 space: real and complex stages may share paths.
    plan = contraction_plan(equation, tuple(a.shape for a in arrays))
    return plan(*arrays)


def ttns_matrix_elements(bras, operator, ket, *, batch_bytes=64*1024**2):
    """Exact <bra|operator|ket> values, grouped by bra tensor shapes.

    All states/operator must share a BasisTree. Bra bond dimensions may
    differ from the ket and from other bras. Grouping avoids degrading an
    entire matrix column to scalar contractions when one vector has a
    different rank. Output order matches ``bras``. Physical indices per node
    may be multiple (including singleton axes). Like TTNS.expectation, this
    contracts tensors without any separate external ``coeff`` factor.

    Stacks are bounded by ``batch_bytes`` per node, with at least one bra;
    contraction intermediates use the usual bounded planner. Live child
    environments, inputs and backend workspace are additional memory.
    The returned array stays on the active NumPy/CuPy backend.
    """
    bras = list(bras)
    if batch_bytes < 1:
        raise ValueError('batch_bytes must be positive')
    if operator.basis is not ket.basis or any(b.basis is not ket.basis for b in bras):
        raise ValueError('matrix elements require a shared BasisTree')
    if not bras:
        return xp.empty(0, dtype=xp.complex128)
    groups = {}
    for index, bra in enumerate(bras):
        shape = tuple(tuple(node.tensor.shape) for node in bra.node_list)
        groups.setdefault(shape, []).append(index)
    result = xp.empty(len(bras), dtype=xp.result_type(
        *[node.tensor.dtype for state in [ket, operator]+bras for node in state.node_list]))
    for indices in groups.values():
        largest = max(node.tensor.size*result.dtype.itemsize for node in bras[indices[0]].node_list)
        chunk = max(1, min(len(indices), batch_bytes//max(1, largest)))
        for start in range(0, len(indices), chunk):
            selected = indices[start:start+chunk]
            env = {}
            for node in ket.postorder_list():
                i = ket.node_idx[node]
                children = [ket.node_idx[c] for c in node.children]
                d = len(children)
                p = node.tensor.ndim-d-1
                b, k, o = [list(range(j*d, (j+1)*d)) for j in range(3)]
                physical_b = list(range(3*d, 3*d+p))
                physical_k = list(range(3*d+p, 3*d+2*p))
                bp, op, kp, batch = range(3*d+2*p, 3*d+2*p+4)
                physical_op = [v for pair in zip(physical_b, physical_k) for v in pair]
                bra_tensor = xp.stack([xp.asarray(bras[j].node_list[i].tensor) for j in selected])
                args = [bra_tensor.conj(), [batch]+b+physical_b+[bp],
                        node.tensor, k+physical_k+[kp],
                        operator.node_list[i].tensor, o+physical_op+[op]]
                for j, child in enumerate(children):
                    args += [env.pop(child), [batch, b[j], o[j], k[j]]]
                env[i] = bounded_contract(*args, [batch, bp, op, kp])
            result[xp.asarray(selected)] = env[ket.node_idx[ket.root]].reshape(len(selected))
    return result


def bounded_local_expression(equation, constants, shape):
    """One dynamic local tensor, with bounded precontraction of constants."""
    shapes = tuple(a.shape for a in constants)+(tuple(shape),)
    plan = contraction_plan(equation, shapes)
    if plan.chunks:
        # Do not cache all sliced constant intermediates simultaneously.
        return lambda value: plan(*constants, value.reshape(shape))
    expression = oe.contract_expression(equation, *constants, shape,
        constants=list(range(len(constants))), optimize=plan.path)
    return lambda value: expression(value.reshape(shape), backend='cupy' if USE_GPU else 'numpy')
