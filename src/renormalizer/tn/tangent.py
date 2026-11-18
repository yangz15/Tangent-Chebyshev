"""One-site tangent vectors and projected TTNO actions for TTNS.

The MPS construction in arXiv:2510.07279 has a natural tree generalization.
With every non-root TTNS tensor isometric towards a fixed root, tangent
components at distinct nodes are orthogonal after imposing ``A^dagger T = 0``.
The branch on the root side carries a non-trivial norm metric.  This module
keeps that metric explicitly and applies a TTNO using ordinary and
one-insertion messages on every directed tree edge.

Local tangent arrays follow Renormalizer's dense-with-structural-zeros tensor
convention.  Admissible coordinates, gauge constraints, parent metrics, and
dimension counts are all resolved by the TTNS bond quantum numbers.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import opt_einsum as oe

from renormalizer.mps.backend import USE_GPU, xp
from renormalizer.mps.matrix import asnumpy, asxp_oe_args
from renormalizer.mps.oe_contract_wrap import oe_contract
from renormalizer.mps.svd_qn import add_outer, get_qn_mask
from renormalizer.tn.tree import TTNEnviron, TTNO
from renormalizer.tn.contractions import contraction_plan


def _sum_arrays(arrays):
    arrays = list(arrays)
    if not arrays:
        raise ValueError("cannot sum an empty collection of tree messages")
    result = arrays[0]
    for array in arrays[1:]:
        result = result + array
    return result


def _run_contraction_task(task):
    owner, order, function = task
    return owner, order, function()


@dataclass
class TTNSTangentVector:
    """A list of gauge-fixed local variations of a reference TTNS."""

    space: "TTNSTangentSpace"
    tensors: List[np.ndarray]

    def copy(self) -> "TTNSTangentVector":
        tensors = self.space._parallel_map(lambda tensor: tensor.copy(), self.tensors)
        return type(self)(self.space, tensors)

    def dot(self, other: "TTNSTangentVector") -> complex:
        self.space._check_vector(other)
        return complex(self.space._host_scalar(self.space._dot_value(self, other)))

    @property
    def norm(self) -> float:
        value = self.dot(self).real
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        return float(np.sqrt(value))

    def scale(self, value, inplace: bool = False) -> "TTNSTangentVector":
        # Multiplication already allocates the output: copying the complete
        # device vector first doubles traffic without protecting anything.
        tensors = self.space._parallel_map(lambda tensor: value * tensor, self.tensors)
        result = self if inplace else type(self)(self.space, tensors)
        result.tensors = tensors
        return result

    def axpy(self, value, other: "TTNSTangentVector") -> "TTNSTangentVector":
        """Perform ``self += value * other`` and return ``self``."""

        self.space._check_vector(other)
        self.tensors = self.space._parallel_map(
            lambda tensors: tensors[0] + value * tensors[1],
            zip(self.tensors, other.tensors),
        )
        return self

    def restore_gauge(self, inplace: bool = False) -> "TTNSTangentVector":
        """Reapply the frozen tangent gauge after cancellation-prone algebra."""

        result = self if inplace else self.copy()
        result.tensors = self.space.project_gauge(result.tensors)
        return result

    def normalize(self) -> "TTNSTangentVector":
        self.restore_gauge(inplace=True)
        norm = self.norm
        if norm == 0.0:
            raise ValueError("cannot normalize a zero tangent vector")
        return self.scale(1.0 / norm, inplace=True)

    def to_dense(self) -> np.ndarray:
        """Materialize the tangent wavefunction (intended for diagnostics)."""

        return self.space.to_dense(self)


class TTNSTangentBasis:
    """Preallocated device-resident tangent basis with batched BLAS algebra.

    A full-reorthogonalized Krylov run repeatedly forms all overlaps with the
    retained basis.  Keeping the basis in node-major arrays turns those
    Python-level vector loops into one GPU GEMV per tree node and turns the
    final Gram matrix into GEMMs.  The class is intentionally GPU-only; the
    existing list implementation remains the lower-overhead CPU path.
    """

    def __init__(self, space: "TTNSTangentSpace", capacity: int, dtype=np.complex128):
        if not space.gpu_resident:
            raise RuntimeError("TTNSTangentBasis requires a GPU-resident tangent space")
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("tangent basis capacity must be a positive integer")
        self.space = space
        self.capacity = capacity
        self.size = 0
        self.tensors = [
            xp.empty((capacity, *shape), dtype=dtype) for shape in space.shapes
        ]

    @property
    def nbytes(self) -> int:
        return int(sum(tensor.nbytes for tensor in self.tensors))

    def append(self, vector: TTNSTangentVector) -> None:
        self.space._check_vector(vector)
        if self.size >= self.capacity:
            raise ValueError("tangent basis capacity exceeded")
        for storage, tensor in zip(self.tensors, vector.tensors):
            storage[self.size] = tensor
        self.size += 1

    def vector(self, index: int) -> TTNSTangentVector:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return TTNSTangentVector(
            self.space, [storage[index] for storage in self.tensors]
        )

    def dot(self, vector: TTNSTangentVector) -> np.ndarray:
        """Return all ``<q_i|vector>`` values with one host transfer."""

        self.space._check_vector(vector)
        result = xp.zeros(self.size, dtype=xp.complex128)
        for storage, right, metric in zip(
            self.tensors, vector.tensors, self.space.parent_metrics
        ):
            parent_dim = right.shape[-1]
            left = storage[: self.size].reshape(self.size, -1, parent_dim)
            right_matrix = right.reshape(-1, parent_dim)
            weighted_right = right_matrix @ metric.T
            # Conjugate the single input vector instead of the entire basis
            # on every reorthogonalization pass: conj(Q) @ v = conj(Q @ conj(v)).
            result += (left.reshape(self.size, -1) @ weighted_right.reshape(-1).conj()).conj()
        return np.asarray(asnumpy(result), dtype=np.complex128)

    def axpy(self, vector: TTNSTangentVector, coefficients) -> TTNSTangentVector:
        """Perform ``vector += sum_i coefficients[i] q_i`` on the GPU."""

        self.space._check_vector(vector)
        coefficients = xp.asarray(coefficients, dtype=xp.complex128)
        if coefficients.shape != (self.size,):
            raise ValueError(
                f"expected {self.size} tangent coefficients, got {coefficients.shape}"
            )
        vector.tensors = [
            tensor
            + xp.tensordot(coefficients, storage[: self.size], axes=(0, 0))
            for tensor, storage in zip(vector.tensors, self.tensors)
        ]
        return vector

    def gram(self) -> np.ndarray:
        """Return the physical tangent Gram matrix using node-wise GEMMs."""

        result = xp.zeros((self.size, self.size), dtype=xp.complex128)
        for storage, metric in zip(self.tensors, self.space.parent_metrics):
            parent_dim = storage.shape[-1]
            matrices = storage[: self.size].reshape(self.size, -1, parent_dim)
            weighted = matrices @ metric.T
            result += (
                matrices.reshape(self.size, -1).conj()
                @ weighted.reshape(self.size, -1).T
            )
        return np.asarray(asnumpy(result), dtype=np.complex128)


class TTNSTangentSpace:
    r"""Root-gauge one-site tangent space of a TTNS.

    Let ``A_v`` be a TTNS tensor viewed as a matrix whose last (parent) bond is
    the column index.  Canonicalization makes ``A_v^dagger A_v = I`` for every
    non-root node.  We use the same discarded-space gauge at every node,

    .. math:: A_v^\dagger T_v=0,

    including the root, where it removes the reference state itself.  The
    tangent-vector Gram matrix is block diagonal.  Its block at node ``v`` is
    the norm message from the root-side branch, stored in
    :attr:`parent_metrics`.

    A projected TTNO action is evaluated with two messages per directed edge:
    an ordinary reference/reference environment and an environment containing
    exactly one tangent insertion.  It is the tree analogue of Eqs. (S8)--(S10)
    of arXiv:2510.07279 and never forms ``TTNO.apply(ttns)``.
    """

    def __init__(
        self,
        reference,
        metric_rcond: float = 1.0e-12,
        target_qntot=None,
        parallel_workers: int = 1,
        gpu_resident: Optional[bool] = None,
        gpu_source_streams: int = 1,
    ):
        if not np.isfinite(metric_rcond) or not 0.0 < metric_rcond < 1.0:
            raise ValueError("metric_rcond must be finite and lie in (0, 1)")
        if not isinstance(parallel_workers, int) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if gpu_resident is None:
            gpu_resident = USE_GPU
        if not isinstance(gpu_resident, bool):
            raise TypeError("gpu_resident must be a bool or None")
        if gpu_resident and not USE_GPU:
            raise RuntimeError(
                "GPU-resident tangent tensors require an initialized CuPy backend; "
                "set RENO_GPU inside a GPU allocation"
            )
        if gpu_resident and parallel_workers != 1:
            raise ValueError(
                "GPU-resident tangent contractions use one host submission thread; "
                "set parallel_workers=1"
            )
        if not isinstance(gpu_source_streams, int) or gpu_source_streams < 1:
            raise ValueError("gpu_source_streams must be a positive integer")
        if gpu_source_streams > 1 and not gpu_resident:
            raise ValueError(
                "gpu_source_streams greater than one requires gpu_resident=True"
            )
        self.metric_rcond = float(metric_rcond)
        self.parallel_workers = int(parallel_workers)
        self.gpu_resident = gpu_resident
        self.array_module = xp if gpu_resident else np
        self._executor = None
        self.gpu_source_streams = int(gpu_source_streams)
        self._gpu_source_executor = None
        self._gpu_stream_pool = ()
        self._contract_path_cache = {}
        self._contract_expression_cache = {}
        self._contract_path_lock = threading.Lock()
        # Strong references in each value prevent object-id reuse while the
        # frozen reference/operator/environment arrays are cached on device.
        self._device_array_cache = {}
        self.last_apply_timing_s = {}
        self.apply_timing_s = {
            "calls": 0,
            "messages_to_parent": 0.0,
            "top_down_and_local_contractions": 0.0,
            "metric_application": 0.0,
            "gauge_projection": 0.0,
            "total": 0.0,
        }
        # Operator/reference environments are independent of the tangent
        # vector.  A Chebyshev recurrence applies the same TTNO hundreds or
        # thousands of times, so rebuilding these environments in every
        # matrix-vector product is pure repeated work.
        self._operator_environment_cache = {}

        reference = reference.copy()
        reference.canonicalise()
        reference.normalize("ttns_and_coeff")
        self.reference = reference
        self.shapes = [node.tensor.shape for node in reference]
        self.reference_qntot = np.asarray(reference.qntot, dtype=int).reshape(-1).copy()
        if target_qntot is None:
            target_qntot = self.reference_qntot
        self.target_qntot = np.asarray(target_qntot, dtype=int).reshape(-1).copy()
        if self.target_qntot.shape != self.reference_qntot.shape:
            raise ValueError(
                "target_qntot and the reference TTNS quantum number have "
                "different sizes"
            )
        self.qntot = self.target_qntot
        self.qn_size = int(self.qntot.size)
        self.parent_qn = [
            np.asarray(node.qn, dtype=int).copy() for node in reference
        ]
        self.qn_masks = [
            get_qn_mask(reference.get_qnmat(node)[-1], self.target_qntot)
            for node in reference
        ]
        self.parent_metric_qn_masks = [
            np.all(qn[:, None, :] == qn[None, :, :], axis=-1)
            for qn in self.parent_qn
        ]
        self.gauge_qn_masks = [
            get_qn_mask(
                add_outer(qn, self.reference_qntot - qn),
                self.target_qntot,
            )
            for qn in self.parent_qn
        ]
        self.qn_active_sizes = [
            int(np.count_nonzero(mask)) for mask in self.qn_masks
        ]
        self.gauge_constraint_sizes = [
            int(np.count_nonzero(mask)) for mask in self.gauge_qn_masks
        ]
        for node_index, (shape, mask) in enumerate(zip(self.shapes, self.qn_masks)):
            if mask.shape != shape:
                raise ValueError(
                    f"quantum-number mask at node {node_index} has shape {mask.shape}, "
                    f"expected {shape}"
                )
        if not any(self.qn_active_sizes):
            raise ValueError(
                f"target total quantum number {self.target_qntot.tolist()} has no tangent coordinates"
            )

        # The dummy TTNO contracts physical indices directly and therefore
        # provides exactly the tangent-coordinate norm metric.
        norm_operator = TTNO.dummy(reference.basis)
        norm_environment = TTNEnviron(reference, norm_operator)
        parent_metrics_host: List[np.ndarray] = []
        parent_metric_pinv_host: List[np.ndarray] = []
        parent_support_host = []
        self.supported_dimensions = []
        for node_index, node in enumerate(reference):
            raw_metric = asnumpy(norm_environment.node_list[node_index].environ_parent)
            metric = raw_metric[:, 0, :]
            metric = 0.5 * (metric + metric.conj().T)
            block_mask = self.parent_metric_qn_masks[node_index]
            metric[~block_mask] = 0

            # Pseudoinvert each parent-charge block independently.  A dense
            # eigensolver may otherwise rotate exactly degenerate eigenvectors
            # between disconnected charge sectors.
            blocks = []
            scale = 1.0
            _, inverse_labels = np.unique(
                self.parent_qn[node_index], axis=0, return_inverse=True
            )
            for label in range(int(np.max(inverse_labels)) + 1):
                indices = np.flatnonzero(inverse_labels == label)
                block = metric[np.ix_(indices, indices)]
                eigenvalues, eigenvectors = np.linalg.eigh(block)
                if eigenvalues.size:
                    scale = max(scale, float(np.max(np.abs(eigenvalues))))
                blocks.append((indices, eigenvalues, eigenvectors))

            cutoff = self.metric_rcond * scale
            pinv = np.zeros_like(metric)
            support = np.zeros_like(metric)
            support_rank = 0
            local_dimension = 0
            active_per_column = self.qn_masks[node_index].reshape(
                -1, metric.shape[0]
            ).sum(axis=0)
            constraints_per_column = self.gauge_qn_masks[node_index].sum(axis=0)
            for indices, eigenvalues, eigenvectors in blocks:
                inverse = np.zeros_like(eigenvalues)
                retained = eigenvalues > cutoff
                inverse[retained] = 1.0 / eigenvalues[retained]
                pinv[np.ix_(indices, indices)] = (
                    eigenvectors * inverse
                ) @ eigenvectors.conj().T
                kept = eigenvectors[:, retained]
                support[np.ix_(indices, indices)] = kept @ kept.conj().T
                rank = int(np.count_nonzero(retained))
                support_rank += rank
                # Every column of a parent-charge block has the same
                # admissible rows and discarded-space gauge constraints.
                local_dimension += rank * int(
                    active_per_column[indices[0]] - constraints_per_column[indices[0]]
                )
            parent_metrics_host.append(metric)
            parent_metric_pinv_host.append(pinv)
            # Skip multiplication by a numerically reconstructed identity
            # when the metric has full support.
            parent_support_host.append(
                None if support_rank == metric.shape[0] else support
            )
            self.supported_dimensions.append(local_dimension)

        self.parent_metrics = [
            self._backend_array(metric, cache=False) for metric in parent_metrics_host
        ]
        self.parent_metric_pinv = [
            self._backend_array(metric, cache=False)
            for metric in parent_metric_pinv_host
        ]
        self.parent_metric_support = [
            None if support is None else self._backend_array(support, cache=False)
            for support in parent_support_host
        ]

        # Nodes at a common height (bottom-up) or depth (top-down) are mutually
        # independent.  Precomputing these wavefronts avoids rediscovering the
        # task graph during every Chebyshev matrix-vector product.
        heights = {}
        postorder_levels = {}
        for node in self.reference.postorder_list():
            height = 0 if not node.children else 1 + max(
                heights[child] for child in node.children
            )
            heights[node] = height
            if node.parent is not None:
                postorder_levels.setdefault(height, []).append(node)
        self._postorder_levels = [
            postorder_levels[level] for level in sorted(postorder_levels)
        ]

        preorder_levels = {}
        for node in self.reference.preorder_list():
            depth = 0 if node.parent is None else len(node.ancestors) - 1
            preorder_levels.setdefault(depth, []).append(node)
        self._preorder_levels = [
            preorder_levels[level] for level in sorted(preorder_levels)
        ]
        self._subtree_nodes = {}
        for node in self.reference.postorder_list():
            self._subtree_nodes[node] = (
                node,
                *(
                    descendant
                    for child in node.children
                    for descendant in self._subtree_nodes[child]
                ),
            )
        all_nodes = tuple(self.reference)
        self._subtree_node_sets = {
            node: frozenset(nodes) for node, nodes in self._subtree_nodes.items()
        }
        self._outside_subtree_nodes = {
            node: tuple(
                other
                for other in all_nodes
                if other not in self._subtree_node_sets[node]
            )
            for node in all_nodes
        }
        self._reference_tensors = [
            self._backend_array(node.tensor) for node in self.reference
        ]
        self._reference_conjugates = [
            tensor.conj() for tensor in self._reference_tensors
        ]
        self._qn_masks_backend = [
            self._backend_array(mask, cache=False) for mask in self.qn_masks
        ]
        self._gauge_qn_masks_backend = [
            self._backend_array(mask, cache=False) for mask in self.gauge_qn_masks
        ]

        if self.parallel_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="renormalizer-tangent",
            )
        self._configure_gpu_source_streams(self.gpu_source_streams)

    def _parallel_map(self, function, values):
        """Apply ``function`` in stable input order using the persistent pool."""

        values = list(values)
        if self._executor is None or len(values) < 2:
            return [function(value) for value in values]
        return list(self._executor.map(function, values))

    def _configure_gpu_source_streams(self, stream_count: int) -> None:
        if self._gpu_source_executor is not None:
            self._gpu_source_executor.shutdown(wait=True)
            self._gpu_source_executor = None
        self._gpu_stream_pool = ()
        self.gpu_source_streams = int(stream_count)
        if not self.gpu_resident or stream_count == 1:
            return
        self._gpu_stream_pool = tuple(
            xp.cuda.Stream(non_blocking=True) for _ in range(stream_count)
        )
        self._gpu_source_executor = ThreadPoolExecutor(
            max_workers=stream_count,
            thread_name_prefix="renormalizer-tangent-gpu-source",
        )

    def set_gpu_source_streams(self, stream_count: int) -> None:
        """Set bounded S5 source concurrency on a GPU-resident space.

        Each CUDA stream has one persistent host scheduling thread and runs
        one source tangent :math:`T^\\ell` at a time.  Sources are distributed
        deterministically across streams and accumulated before the final
        stream-order reduction, so memory grows with ``stream_count`` rather
        than with the number of tangent components.
        """

        if not isinstance(stream_count, int) or stream_count < 1:
            raise ValueError("stream_count must be a positive integer")
        if not self.gpu_resident and stream_count != 1:
            raise ValueError("multiple GPU source streams require gpu_resident=True")
        if stream_count == self.gpu_source_streams:
            return
        self.synchronize()
        self._configure_gpu_source_streams(stream_count)

    def _gpu_stream_reduce_source_tasks(self, function, tasks):
        """Run exact S5 source traversals with one active ell per stream."""

        tasks = list(tasks)
        if not tasks:
            return []
        if not self.gpu_resident or self.gpu_source_streams == 1:
            return self._parallel_map(function, tasks)

        # All frozen arrays and the input tangent are prepared on the default
        # stream.  Make them visible before non-blocking streams consume them.
        self.synchronize()
        stream_count = min(self.gpu_source_streams, len(tasks))
        buckets = [tasks[index::stream_count] for index in range(stream_count)]
        device_id = int(xp.cuda.Device().id)

        def run_bucket(payload):
            stream_index, bucket = payload
            stream = self._gpu_stream_pool[stream_index]
            accumulated = None
            with xp.cuda.Device(device_id), stream:
                for task in bucket:
                    outputs = function(task)
                    # A source traversal owns many temporary directed-edge
                    # messages.  Complete it before those arrays are released
                    # to CuPy's shared memory pool and potentially reused by a
                    # different stream.
                    stream.synchronize()
                    if accumulated is None:
                        accumulated = outputs
                    else:
                        for target_index, output in enumerate(outputs):
                            accumulated[target_index] += output
                        stream.synchronize()
            return accumulated

        return list(
            self._gpu_source_executor.map(
                run_bucket, enumerate(buckets)
            )
        )

    def _backend_array(self, array, *, cache: bool = True):
        """Return an array on the tangent backend, caching frozen GPU constants."""

        if not self.gpu_resident:
            return np.asarray(array)
        if isinstance(array, xp.ndarray):
            return array
        if not cache:
            return xp.asarray(array)
        key = id(array)
        cached = self._device_array_cache.get(key)
        if cached is not None and cached[0] is array:
            return cached[1]
        converted = xp.asarray(array)
        self._device_array_cache[key] = (array, converted)
        return converted

    def _backend_oe_args(self, arguments):
        converted = []
        for position, argument in enumerate(arguments[:-1]):
            converted.append(
                self._backend_array(argument) if position % 2 == 0 else argument
            )
        converted.append(arguments[-1])
        return converted

    def _host_scalar(self, value):
        if self.gpu_resident:
            value = asnumpy(value)
        return np.asarray(value).reshape(()).item()

    def _dot_value(self, left: TTNSTangentVector, right: TTNSTangentVector):
        values = []
        for left_tensor, right_tensor, metric in zip(
            left.tensors, right.tensors, self.parent_metrics
        ):
            parent_dim = left_tensor.shape[-1]
            left_matrix = left_tensor.reshape(-1, parent_dim)
            right_matrix = right_tensor.reshape(-1, parent_dim)
            values.append(
                self.array_module.einsum(
                    "pr,ps,rs->",
                    left_matrix.conj(),
                    right_matrix,
                    metric,
                    optimize=True,
                )
            )
        # Stable node-order reduction matches the CPU implementation.
        return sum(values, 0.0j)

    def synchronize(self) -> None:
        if self.gpu_resident:
            xp.cuda.get_current_stream().synchronize()

    @property
    def backend_name(self) -> str:
        if not self.gpu_resident:
            return "numpy"
        properties = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        name = properties.get("name", b"GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return f"cupy:{name}"

    @property
    def device_cache_nbytes(self) -> int:
        if not self.gpu_resident:
            return 0
        return int(sum(value[1].nbytes for value in self._device_array_cache.values()))

    def allocate_basis(
        self, capacity: int, dtype=np.complex128
    ) -> TTNSTangentBasis:
        return TTNSTangentBasis(self, capacity, dtype=dtype)

    def set_parallel_workers(self, parallel_workers: int) -> None:
        """Resize the persistent CPU pool between tangent operations.

        This is primarily useful for strong-scaling measurements: all worker
        counts then reuse the same frozen tangent metrics, TTNO environments,
        and contraction paths.
        """

        if not isinstance(parallel_workers, int) or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if self.gpu_resident and parallel_workers != 1:
            raise ValueError(
                "GPU-resident tangent contractions use one host submission thread"
            )
        if parallel_workers == self.parallel_workers and (
            parallel_workers == 1 or self._executor is not None
        ):
            return
        self.shutdown()
        self.parallel_workers = int(parallel_workers)
        if self.parallel_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.parallel_workers,
                thread_name_prefix="renormalizer-tangent",
            )

    def shutdown(self, wait: bool = True) -> None:
        """Release CPU worker threads owned by this tangent space."""

        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
        if self._gpu_source_executor is not None:
            self._gpu_source_executor.shutdown(wait=wait)
            self._gpu_source_executor = None
        self._gpu_stream_pool = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    @property
    def cached_contraction_paths(self) -> int:
        return len(self._contract_path_cache)

    @property
    def node_num(self) -> int:
        return len(self.reference)

    @property
    def dimension(self) -> int:
        return int(sum(self.supported_dimensions))

    @property
    def quantum_number_aware(self) -> bool:
        """Whether local coordinates enforce the selected target sector."""

        return True

    def _check_vector(self, vector: TTNSTangentVector) -> None:
        if not isinstance(vector, TTNSTangentVector) or vector.space is not self:
            raise ValueError("tangent vectors belong to different TTNS tangent spaces")
        if len(vector.tensors) != self.node_num:
            raise ValueError("wrong number of tangent tensors")
        invalid = []
        for node_index, (tensor, shape) in enumerate(zip(vector.tensors, self.shapes)):
            if tensor.shape != shape:
                raise ValueError(
                    f"tangent tensor {node_index} has shape {tensor.shape}, expected {shape}"
                )
            forbidden = tensor[~self._qn_masks_backend[node_index]]
            if forbidden.size:
                invalid.append(self.array_module.any(forbidden != 0))
        # One device synchronization per validation, rather than one per
        # charged node. Keep the public structural-zero check intact.
        if invalid and bool(self._host_scalar(self.array_module.any(self.array_module.stack(invalid)))):
            raise ValueError(
                "tangent tensor contains entries outside total-charge "
                f"sector {self.target_qntot.tolist()}"
            )

    def _operator_environment(self, ttno) -> TTNEnviron:
        """Return the immutable reference/TTNO environment for ``ttno``."""

        key = id(ttno)
        cached = self._operator_environment_cache.get(key)
        if cached is not None and cached[0] is ttno:
            return cached[1]
        environment = TTNEnviron(self.reference, ttno)
        # Retaining the TTNO alongside its environment prevents Python from
        # reusing an id while the cache entry is live.
        self._operator_environment_cache[key] = (ttno, environment)
        return environment

    def _mask_tensors(self, tensors: Iterable[np.ndarray]) -> List[np.ndarray]:
        tensors = list(tensors)
        if len(tensors) != self.node_num:
            raise ValueError("wrong number of tangent tensors")
        output = []
        for node_index, (tensor, shape, mask) in enumerate(
            zip(tensors, self.shapes, self._qn_masks_backend)
        ):
            tensor = self._backend_array(tensor, cache=False)
            if tensor.shape != shape:
                raise ValueError(
                    f"tangent tensor {node_index} has shape {tensor.shape}, expected {shape}"
                )
            masked = self.array_module.zeros(shape, dtype=tensor.dtype)
            masked[mask] = tensor[mask]
            support = self.parent_metric_support[node_index]
            if support is not None:
                # The pseudoinverse used by apply() acts only on this
                # Schmidt support. All vectors, including the source and
                # project=False coordinates, must live in that same space;
                # otherwise P I P is not the identity in a recurrence.
                masked = (masked.reshape(-1, shape[-1]) @ support.T).reshape(shape)
                masked[~mask] = 0
            output.append(masked)
        return output

    def project_gauge(self, tensors: Iterable[np.ndarray]) -> List[np.ndarray]:
        tensors = self._mask_tensors(tensors)
        def project(values):
            tensor, isometry_tensor, shape, mask, gauge_mask = values
            parent_dim = shape[-1]
            isometry = isometry_tensor.reshape(-1, parent_dim)
            matrix = tensor.reshape(-1, parent_dim)
            overlap = isometry.conj().T @ matrix
            overlap[~gauge_mask] = 0
            matrix = matrix - isometry @ overlap
            result = matrix.reshape(shape)
            result[~mask] = 0
            return result

        return self._parallel_map(
            project,
            zip(
                tensors,
                self._reference_tensors,
                self.shapes,
                self._qn_masks_backend,
                self._gauge_qn_masks_backend,
            ),
        )

    def vector(
        self, tensors: Iterable[np.ndarray], project: bool = True
    ) -> TTNSTangentVector:
        if project:
            tensors = self.project_gauge(tensors)
        else:
            tensors = self._mask_tensors(tensors)
        vector = TTNSTangentVector(self, tensors)
        self._check_vector(vector)
        return vector

    def zeros(self, dtype=None) -> TTNSTangentVector:
        if dtype is None:
            dtype = np.result_type(*[node.tensor.dtype for node in self.reference])
        return self.vector(
            [self.array_module.zeros(shape, dtype=dtype) for shape in self.shapes]
        )

    def random(
        self, seed: Optional[int] = None, complex_values: Optional[bool] = None
    ) -> TTNSTangentVector:
        rng = np.random.default_rng(seed)
        if complex_values is None:
            complex_values = any(
                np.iscomplexobj(node.tensor) for node in self.reference
            )
        tensors = []
        for shape, mask in zip(self.shapes, self.qn_masks):
            tensor = np.zeros(shape)
            tensor[mask] = rng.standard_normal(np.count_nonzero(mask))
            if complex_values:
                imaginary = np.zeros(shape)
                imaginary[mask] = rng.standard_normal(np.count_nonzero(mask))
                tensor = tensor + 1j * imaginary
            tensors.append(tensor)
        return self.vector(tensors).normalize()

    def gauge_residual(self, vector: TTNSTangentVector) -> float:
        self._check_vector(vector)
        residual = 0.0
        for isometry_tensor, tensor in zip(self._reference_tensors, vector.tensors):
            parent_dim = isometry_tensor.shape[-1]
            isometry = isometry_tensor.reshape(-1, parent_dim)
            matrix = tensor.reshape(-1, parent_dim)
            residual = max(
                residual,
                float(
                    self._host_scalar(
                        self.array_module.linalg.norm(isometry.conj().T @ matrix)
                    )
                ),
            )
        return residual

    def sector_residual(self, vector: TTNSTangentVector) -> float:
        """Return the largest norm outside the selected total-charge sector."""

        self._check_vector(vector)
        return max(
            (
                float(
                    self._host_scalar(
                        self.array_module.linalg.norm(tensor[~mask])
                    )
                )
                for tensor, mask in zip(vector.tensors, self._qn_masks_backend)
            ),
            default=0.0,
        )

    def _contract_local(
        self,
        ttno,
        environment,
        node,
        ket_tensor,
        child_messages,
        parent_message,
        *,
        omit_child=None,
        omit_parent=False,
        include_bra=True,
        output="parent",
    ):
        """Contract a local tree network with selectable incoming messages."""

        node_index = self.reference.node_idx[node]
        enode = environment.node_list[node_index]
        operator_node = ttno.node_list[node_index]
        args = []

        for child_index, message in enumerate(child_messages):
            if child_index == omit_child:
                continue
            args.extend(
                [
                    message,
                    environment.get_child_indices(
                        enode, child_index, self.reference, ttno
                    ),
                ]
            )
        if not omit_parent:
            args.extend(
                [
                    parent_message,
                    environment.get_parent_indices(enode, self.reference, ttno),
                ]
            )
        if include_bra:
            args.extend(
                [
                    self._reference_conjugates[node_index],
                    self.reference.get_node_indices(node, conj=True),
                ]
            )
        args.extend([operator_node.tensor, ttno.get_node_indices(operator_node)])
        args.extend(
            [ket_tensor, self.reference.get_node_indices(node, ttno=ttno)]
        )

        if output == "parent":
            output_indices = environment.get_parent_indices(
                enode, self.reference, ttno
            )
        elif output == "child":
            if omit_child is None:
                raise ValueError("child output requires omit_child")
            output_indices = environment.get_child_indices(
                enode, omit_child, self.reference, ttno
            )
        elif output == "bra":
            output_indices = self.reference.get_node_indices(node, conj=True)
        else:
            raise ValueError(f"unknown tree-message output {output!r}")
        args.append(output_indices)
        return self._cached_contract(args)

    @staticmethod
    def _contraction_key(arguments):
        """Normalize labels so equivalent contractions share one cached path."""

        label_numbers = {}

        def normalize(indices):
            normalized = []
            for label in indices:
                try:
                    number = label_numbers[label]
                except KeyError:
                    number = len(label_numbers)
                    label_numbers[label] = number
                normalized.append(number)
            return tuple(normalized)

        operands = []
        for position in range(0, len(arguments) - 1, 2):
            operands.append(
                (tuple(arguments[position].shape), normalize(arguments[position + 1]))
            )
        return tuple(operands), normalize(arguments[-1])

    def _cached_contract(self, arguments):
        key = self._contraction_key(arguments)
        expression = self._contract_expression_cache.get(key)
        if expression is None:
            # Path generation is pure setup work for a frozen reference/TTNO.
            # Serialize only the first encounter; contractions themselves run
            # without this lock and therefore execute concurrently.
            with self._contract_path_lock:
                expression = self._contract_expression_cache.get(key)
                if expression is None:
                    # Reuse the parsed contraction program as well as its
                    # path. Every operand is supplied afresh; no tangent
                    # insertion or message is ever cached as a constant.
                    operands, output = key
                    equation = ','.join(''.join(oe.get_symbol(i) for i in indices)
                                        for _, indices in operands)
                    equation += '->'+''.join(oe.get_symbol(i) for i in output)
                    expression = contraction_plan(
                        equation, tuple(shape for shape, _ in operands))
                    self._contract_path_cache[key] = expression.path
                    self._contract_expression_cache[key] = expression
        if self.gpu_resident:
            converted = self._backend_oe_args(arguments)
            return expression(*converted[:-1:2], backend='cupy')
        converted = asxp_oe_args(arguments)
        return asnumpy(expression(*converted[:-1:2], backend='cupy' if USE_GPU else 'numpy'))

    def _evaluate_contraction_groups(self, groups):
        """Evaluate independent contractions and sum each group stably."""

        tasks = [
            (owner, order, function)
            for owner, functions in groups
            for order, function in enumerate(functions)
        ]
        grouped = {
            owner: [None] * len(functions) for owner, functions in groups
        }
        for owner, order, value in self._parallel_map(
            _run_contraction_task, tasks
        ):
            grouped[owner][order] = value
        return {
            owner: _sum_arrays(grouped[owner]) for owner, _functions in groups
        }

    def _evaluate_contraction_dag(self, initial_groups, group_completed):
        """Evaluate dynamically released contraction groups.

        ``group_completed(owner, value)`` runs on the calling thread after a
        group's values have been reduced in their original order.  It may
        return more groups whose dependencies are now satisfied.  Individual
        contractions run in the persistent executor, so a ready parent or
        child can start without waiting for an unrelated tree-depth barrier.
        """

        initial_groups = list(initial_groups)
        if self._executor is None:
            queue = initial_groups
            cursor = 0
            while cursor < len(queue):
                owner, functions = queue[cursor]
                cursor += 1
                value = _sum_arrays(function() for function in functions)
                released = group_completed(owner, value)
                if released:
                    queue.extend(released)
            return

        results = {}
        remaining = {}
        future_metadata = {}

        def submit(groups):
            for owner, functions in groups:
                functions = list(functions)
                if not functions:
                    raise ValueError("a contraction DAG group cannot be empty")
                if owner in results:
                    raise ValueError(f"duplicate contraction DAG owner {owner!r}")
                results[owner] = [None] * len(functions)
                remaining[owner] = len(functions)
                for order, function in enumerate(functions):
                    future = self._executor.submit(function)
                    future_metadata[future] = (owner, order)

        submit(initial_groups)
        try:
            while future_metadata:
                completed_futures, _pending = wait(
                    tuple(future_metadata), return_when=FIRST_COMPLETED
                )
                for future in completed_futures:
                    owner, order = future_metadata.pop(future)
                    results[owner][order] = future.result()
                    remaining[owner] -= 1
                    if remaining[owner] != 0:
                        continue
                    value = _sum_arrays(results.pop(owner))
                    del remaining[owner]
                    released = group_completed(owner, value)
                    if released:
                        submit(released)
        except BaseException:
            for future in future_metadata:
                future.cancel()
            raise

    def _apply_sum_by_tangent_components_levelwise(
        self,
        ttnos,
        environments,
        vector,
        energy_shift,
        started,
    ):
        r"""Apply with separate messages for every input tangent block.

        The ordinary application sums tangent insertions while propagating
        each edge message.  Here the source node label is retained, exposing
        every :math:`P_i H T_j` contraction as an independent task.  This uses
        more arithmetic and memory but provides an intentionally wide task
        graph for large CPU allocations.
        """

        inserted_to_parent = {}

        def call_to_parent(operator_index, node, source):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            if source is node:
                ket_tensor = vector.tensors[node_index]
            else:
                ket_tensor = node.tensor
                for child_index, child in enumerate(node.children):
                    if source in self._subtree_node_sets[child]:
                        base_children[child_index] = inserted_to_parent[
                            (operator_index, source, child)
                        ]
                        break
                else:  # pragma: no cover - guarded by the source iteration
                    raise RuntimeError("tangent source is outside the node subtree")
            return partial(
                self._contract_local,
                ttno,
                environment,
                node,
                ket_tensor,
                base_children,
                enode.environ_parent,
                omit_parent=True,
                output="parent",
            )

        for level in self._postorder_levels:
            groups = [
                (
                    (operator_index, source, node),
                    [call_to_parent(operator_index, node, source)],
                )
                for operator_index in range(len(ttnos))
                for node in level
                for source in self._subtree_nodes[node]
            ]
            inserted_to_parent.update(self._evaluate_contraction_groups(groups))
        to_parent_seconds = time.perf_counter() - started

        inserted_from_parent = {}

        def call_from_parent(operator_index, node, child_index, child, source):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            if source is node:
                ket_tensor = vector.tensors[node_index]
                parent_message = enode.environ_parent
            else:
                ket_tensor = node.tensor
                parent_message = enode.environ_parent
                found_sibling = False
                for sibling_index, sibling in enumerate(node.children):
                    if sibling_index == child_index:
                        continue
                    if source in self._subtree_node_sets[sibling]:
                        base_children[sibling_index] = inserted_to_parent[
                            (operator_index, source, sibling)
                        ]
                        found_sibling = True
                        break
                if not found_sibling:
                    parent_message = inserted_from_parent[
                        (operator_index, source, node)
                    ]
            return partial(
                self._contract_local,
                ttno,
                environment,
                node,
                ket_tensor,
                base_children,
                parent_message,
                omit_child=child_index,
                output="child",
            )

        parent_started = time.perf_counter()
        for level in self._preorder_levels:
            groups = [
                (
                    (operator_index, source, child),
                    [
                        call_from_parent(
                            operator_index,
                            node,
                            child_index,
                            child,
                            source,
                        )
                    ],
                )
                for operator_index in range(len(ttnos))
                for node in level
                for child_index, child in enumerate(node.children)
                for source in self._outside_subtree_nodes[child]
            ]
            inserted_from_parent.update(
                self._evaluate_contraction_groups(groups)
            )
        messages_from_parent_seconds = time.perf_counter() - parent_started

        def output_call(operator_index, node, source):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            parent_message = enode.environ_parent
            if source is node:
                ket_tensor = vector.tensors[node_index]
            else:
                ket_tensor = node.tensor
                for child_index, child in enumerate(node.children):
                    if source in self._subtree_node_sets[child]:
                        base_children[child_index] = inserted_to_parent[
                            (operator_index, source, child)
                        ]
                        break
                else:
                    parent_message = inserted_from_parent[
                        (operator_index, source, node)
                    ]
            return partial(
                self._contract_local,
                ttno,
                environment,
                node,
                ket_tensor,
                base_children,
                parent_message,
                include_bra=False,
                output="bra",
            )

        projector_started = time.perf_counter()
        output_cotangents = self._evaluate_contraction_groups(
            [
                (
                    node,
                    [
                        output_call(operator_index, node, source)
                        for operator_index in range(len(ttnos))
                        for source in self.reference
                    ],
                )
                for node in self.reference
            ]
        )
        local_projectors_seconds = time.perf_counter() - projector_started
        top_down_output_seconds = (
            messages_from_parent_seconds + local_projectors_seconds
        )

        def finish_output(node):
            node_index = self.reference.node_idx[node]
            parent_dim = node.tensor.shape[-1]
            cotangent_matrix = output_cotangents[node].reshape(-1, parent_dim)
            tangent_matrix = (
                cotangent_matrix @ self.parent_metric_pinv[node_index].T
            )
            tangent = tangent_matrix.reshape(node.tensor.shape)
            if energy_shift != 0.0:
                tangent = tangent - energy_shift * vector.tensors[node_index]
            return tangent

        metric_started = time.perf_counter()
        output_tensors = self._parallel_map(finish_output, self.reference)
        metric_seconds = time.perf_counter() - metric_started
        gauge_started = time.perf_counter()
        result = self.vector(output_tensors, project=True)
        gauge_seconds = time.perf_counter() - gauge_started
        self.synchronize()
        total_seconds = time.perf_counter() - started
        self.last_apply_timing_s = {
            "operator_components": len(ttnos),
            "parallel_projectors": False,
            "parallel_tangent_components": True,
            "tangent_component_blocks": self.node_num,
            "messages_to_parent": to_parent_seconds,
            "messages_from_parent": messages_from_parent_seconds,
            "local_projectors": local_projectors_seconds,
            "top_down_and_local_contractions": top_down_output_seconds,
            "metric_application": metric_seconds,
            "gauge_projection": gauge_seconds,
            "total": total_seconds,
        }
        self.apply_timing_s["calls"] += 1
        for name, seconds in self.last_apply_timing_s.items():
            if name in self.apply_timing_s:
                self.apply_timing_s[name] += seconds
        return result

    def _apply_sum_by_tangent_components(
        self,
        ttnos,
        environments,
        vector,
        energy_shift,
        started,
        source_task_partition=None,
    ):
        r"""Apply S5 with one coarse, independent task per source tangent.

        For a fixed source node :math:`j`, the one-insertion message can be
        propagated away from :math:`j` over the tree and used to construct
        every :math:`P_i H T_j`.  Different source nodes never depend on one
        another.  Scheduling the complete traversal as one task removes the
        level barriers and thousands of tiny executor submissions incurred by
        :meth:`_apply_sum_by_tangent_components_levelwise`, while retaining
        the explicit source-node decomposition of Eq. S5.

        Results are returned and reduced in ``(TTNO, source-node)`` order, so
        thread completion order cannot affect the numerical answer.
        """

        nodes = tuple(self.reference)

        def neighbours(node):
            if node.parent is None:
                return tuple(node.children)
            return (node.parent, *node.children)

        def source_traversal(task):
            operator_index, source_index = task
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            source = nodes[source_index]
            directed_messages = {}
            predecessor = {source: None}
            queue = []

            def contract_message(node, previous, destination, ket_tensor):
                node_index = self.reference.node_idx[node]
                enode = environment.node_list[node_index]
                child_messages = list(enode.environ_children)
                parent_message = enode.environ_parent
                if previous is not None:
                    incoming = directed_messages[(previous, node)]
                    if previous is node.parent:
                        parent_message = incoming
                    else:
                        child_messages[node.children.index(previous)] = incoming

                if destination is node.parent:
                    return self._contract_local(
                        ttno,
                        environment,
                        node,
                        ket_tensor,
                        child_messages,
                        parent_message,
                        omit_parent=True,
                        output="parent",
                    )
                child_index = node.children.index(destination)
                return self._contract_local(
                    ttno,
                    environment,
                    node,
                    ket_tensor,
                    child_messages,
                    parent_message,
                    omit_child=child_index,
                    output="child",
                )

            # Emit the insertion from its source in every direction.  A
            # breadth-first traversal then carries it over each directed edge
            # exactly once.
            source_tensor = vector.tensors[source_index]
            for destination in neighbours(source):
                directed_messages[(source, destination)] = contract_message(
                    source, None, destination, source_tensor
                )
                predecessor[destination] = source
                queue.append(destination)

            cursor = 0
            while cursor < len(queue):
                node = queue[cursor]
                cursor += 1
                previous = predecessor[node]
                for destination in neighbours(node):
                    if destination is previous:
                        continue
                    directed_messages[(node, destination)] = contract_message(
                        node, previous, destination, node.tensor
                    )
                    predecessor[destination] = node
                    queue.append(destination)

            outputs = []
            for target_index, target in enumerate(nodes):
                enode = environment.node_list[target_index]
                child_messages = list(enode.environ_children)
                parent_message = enode.environ_parent
                if target is source:
                    ket_tensor = source_tensor
                else:
                    ket_tensor = target.tensor
                    previous = predecessor[target]
                    incoming = directed_messages[(previous, target)]
                    if previous is target.parent:
                        parent_message = incoming
                    else:
                        child_messages[target.children.index(previous)] = incoming
                outputs.append(
                    self._contract_local(
                        ttno,
                        environment,
                        target,
                        ket_tensor,
                        child_messages,
                        parent_message,
                        include_bra=False,
                        output="bra",
                    )
                )
            return outputs

        component_started = time.perf_counter()
        all_source_tasks = [
            (operator_index, source_index)
            for operator_index in range(len(ttnos))
            for source_index in range(len(nodes))
        ]
        if source_task_partition is None:
            partition_rank, partition_size = 0, 1
        else:
            partition_rank, partition_size = source_task_partition
        source_tasks = [
            task
            for task_index, task in enumerate(all_source_tasks)
            if task_index % partition_size == partition_rank
        ]
        source_outputs = self._gpu_stream_reduce_source_tasks(
            source_traversal, source_tasks
        )
        component_seconds = time.perf_counter() - component_started

        # The CPU path preserves operator-major/source-minor task order.  The
        # multi-stream GPU path returns deterministic per-stream partial sums;
        # reducing those in stream order gives the same S5 vector to floating
        # precision while bounding live output storage by the stream count.
        output_cotangents = [
            _sum_arrays(outputs[target_index] for outputs in source_outputs)
            for target_index in range(len(nodes))
        ]

        def finish_output(node):
            node_index = self.reference.node_idx[node]
            parent_dim = node.tensor.shape[-1]
            cotangent_matrix = output_cotangents[node_index].reshape(
                -1, parent_dim
            )
            tangent_matrix = (
                cotangent_matrix @ self.parent_metric_pinv[node_index].T
            )
            tangent = tangent_matrix.reshape(node.tensor.shape)
            if energy_shift != 0.0:
                tangent = tangent - energy_shift * vector.tensors[node_index]
            return tangent

        metric_started = time.perf_counter()
        output_tensors = self._parallel_map(finish_output, nodes)
        metric_seconds = time.perf_counter() - metric_started
        gauge_started = time.perf_counter()
        result = self.vector(output_tensors, project=True)
        gauge_seconds = time.perf_counter() - gauge_started
        self.synchronize()
        total_seconds = time.perf_counter() - started
        self.last_apply_timing_s = {
            "operator_components": len(ttnos),
            "parallel_projectors": False,
            "parallel_tangent_components": True,
            "tangent_component_schedule": (
                "gpu_stream_source_tasks"
                if self.gpu_resident and self.gpu_source_streams > 1
                else "source_tasks"
            ),
            "tangent_component_blocks": self.node_num,
            "source_task_count": len(source_tasks),
            "global_source_task_count": len(all_source_tasks),
            "source_task_partition_rank": partition_rank,
            "source_task_partition_size": partition_size,
            "gpu_source_streams": (
                self.gpu_source_streams if self.gpu_resident else 0
            ),
            "source_traversals": component_seconds,
            "messages_to_parent": 0.0,
            "messages_from_parent": 0.0,
            "local_projectors": component_seconds,
            "top_down_and_local_contractions": component_seconds,
            "metric_application": metric_seconds,
            "gauge_projection": gauge_seconds,
            "total": total_seconds,
        }
        self.apply_timing_s["calls"] += 1
        for name, seconds in self.last_apply_timing_s.items():
            if name in self.apply_timing_s:
                self.apply_timing_s[name] += seconds
        return result

    def apply(
        self,
        ttno,
        vector: TTNSTangentVector,
        energy_shift: complex = 0.0,
        *,
        parallel_projectors: bool = False,
        parallel_tangent_components: bool = False,
        tree_schedule: str = "auto",
        source_task_partition=None,
    ) -> TTNSTangentVector:
        r"""Return :math:`P^{1\perp}(H-E)P^{1\perp}|\Phi\rangle`.

        This is the single-operator form of :meth:`apply_sum`.
        """

        return self.apply_sum(
            (ttno,),
            vector,
            energy_shift=energy_shift,
            parallel_projectors=parallel_projectors,
            parallel_tangent_components=parallel_tangent_components,
            tree_schedule=tree_schedule,
            source_task_partition=source_task_partition,
        )

    def apply_sum(
        self,
        ttnos,
        vector: TTNSTangentVector,
        energy_shift: complex = 0.0,
        *,
        parallel_projectors: bool = False,
        parallel_tangent_components: bool = False,
        tree_schedule: str = "auto",
        source_task_partition=None,
    ) -> TTNSTangentVector:
        r"""Apply a sum of TTNOs inside the frozen tangent space.

        Each TTNO contribution has its own independent tree-message graph.
        Message contractions from all operator components are placed in one
        shared task wavefront, exposing Hamiltonian-term parallelism in
        addition to the node parallelism of :meth:`apply`.  Cotangents are
        reduced in the input TTNO order before the tangent metric, energy
        shift, and gauge projection are applied once.  The reduction order is
        independent of thread scheduling.

        The result uses the Riesz representation associated with each parent
        metric. Directions below ``metric_rcond`` are excluded both from
        tangent coordinates and from the pseudoinverse. Discarding a small
        nonzero Schmidt direction is an additional numerical approximation.

        When ``parallel_projectors`` is true, the root-side insertion messages
        are completed first and every local output projector is then placed in
        one global task wavefront.  This realizes the orthogonal decomposition
        :math:`P^{1\perp}=\sum_v P_v` without rebuilding the shared insertion
        environments separately for every tangent component.
        """

        self._check_vector(vector)
        if not isinstance(parallel_projectors, bool):
            raise TypeError("parallel_projectors must be a bool")
        if not isinstance(parallel_tangent_components, bool):
            raise TypeError("parallel_tangent_components must be a bool")
        if parallel_projectors and parallel_tangent_components:
            raise ValueError(
                "parallel_projectors and parallel_tangent_components are "
                "alternative schedules"
            )
        if source_task_partition is not None:
            if not parallel_tangent_components:
                raise ValueError(
                    "source_task_partition requires "
                    "parallel_tangent_components=True"
                )
            if (
                not isinstance(source_task_partition, tuple)
                or len(source_task_partition) != 2
                or not all(isinstance(value, int) for value in source_task_partition)
            ):
                raise TypeError(
                    "source_task_partition must be a (rank, size) integer tuple"
                )
            partition_rank, partition_size = source_task_partition
            if partition_size < 1 or not 0 <= partition_rank < partition_size:
                raise ValueError(
                    "source_task_partition must satisfy "
                    "0 <= rank < size and size >= 1"
                )
        if tree_schedule not in {"auto", "dependency_dag", "level_wavefront"}:
            raise ValueError(
                "tree_schedule must be 'auto', 'dependency_dag', or "
                "'level_wavefront'"
            )
        ttnos = tuple(ttnos)
        if not ttnos:
            raise ValueError("apply_sum requires at least one TTNO")
        for ttno in ttnos:
            TTNEnviron._assert_compatible_basis(self.reference.basis, ttno.basis)
            operator_qntot = np.asarray(ttno.qntot, dtype=int).reshape(-1)
            if operator_qntot.shape != self.qntot.shape or np.any(operator_qntot != 0):
                raise ValueError(
                    "a tangent-space endomorphism must use charge-neutral TTNOs; "
                    f"received operator quantum number {operator_qntot.tolist()}"
                )
        started = time.perf_counter()
        # Build/cache environments before entering parallel contraction
        # wavefronts.  This keeps the cache mutation serial and makes repeated
        # H_T psi benchmarks measure the application rather than setup.
        environments = tuple(self._operator_environment(ttno) for ttno in ttnos)
        if parallel_tangent_components:
            return self._apply_sum_by_tangent_components(
                ttnos,
                environments,
                vector,
                energy_shift if source_task_partition in (None, (0, 1))
                or source_task_partition[0] == 0 else 0.0,
                started,
                source_task_partition=source_task_partition,
            )

        # Sum of all tangent insertions in each child-side branch, propagated
        # once from leaves to root.
        inserted_to_parent: Dict[object, np.ndarray] = {}

        def calls_to_parent(operator_index, node):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            calls = [
                partial(
                    self._contract_local,
                    ttno,
                    environment,
                    node,
                    vector.tensors[node_index],
                    base_children,
                    enode.environ_parent,
                    omit_parent=True,
                    output="parent",
                )
            ]
            for child_index, child in enumerate(node.children):
                messages = base_children.copy()
                messages[child_index] = inserted_to_parent[(operator_index, child)]
                calls.append(
                    partial(
                        self._contract_local,
                        ttno,
                        environment,
                        node,
                        node.tensor,
                        messages,
                        enode.environ_parent,
                        omit_parent=True,
                        output="parent",
                    )
                )
            return calls

        # ``auto`` preserves the production choice: serial level traversal at
        # one worker and the asynchronous recursive S9 DAG once a worker pool
        # exists.  The explicit choices are intentionally public so an
        # autotuner can compare scheduling alone on the same frozen tangent
        # space, environments, and contraction paths.
        dependency_dag = tree_schedule == "dependency_dag" or (
            tree_schedule == "auto" and self._executor is not None
        )
        if dependency_dag:
            scheduled_to_parent = set()

            def to_parent_group(operator_index, node):
                owner = (operator_index, node)
                scheduled_to_parent.add(owner)
                return owner, calls_to_parent(operator_index, node)

            initial_groups = [
                to_parent_group(operator_index, node)
                for operator_index in range(len(ttnos))
                for node in self.reference
                if node.parent is not None and not node.children
            ]

            def to_parent_completed(owner, value):
                operator_index, node = owner
                inserted_to_parent[owner] = value
                parent = node.parent
                parent_owner = (operator_index, parent)
                if (
                    parent.parent is not None
                    and parent_owner not in scheduled_to_parent
                    and all(
                        (operator_index, child) in inserted_to_parent
                        for child in parent.children
                    )
                ):
                    return [to_parent_group(operator_index, parent)]
                return ()

            self._evaluate_contraction_dag(
                initial_groups, to_parent_completed
            )
        else:
            for level in self._postorder_levels:
                groups = [
                    (
                        (operator_index, node),
                        calls_to_parent(operator_index, node),
                    )
                    for operator_index in range(len(ttnos))
                    for node in level
                ]
                inserted_to_parent.update(
                    self._evaluate_contraction_groups(groups)
                )
        to_parent_seconds = time.perf_counter() - started

        # Sum of insertions in each root-side branch, propagated from root to
        # leaves.  For a child c, the branch contains the parent node, all of
        # its other children, and (except at the root) its own parent branch.
        inserted_from_parent: Dict[object, np.ndarray] = {}

        def calls_from_parent(operator_index, item):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node, output_child_index, output_child = item
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            calls = [
                partial(
                    self._contract_local,
                    ttno,
                    environment,
                    node,
                    vector.tensors[node_index],
                    base_children,
                    enode.environ_parent,
                    omit_child=output_child_index,
                    output="child",
                )
            ]
            if node.parent is not None:
                calls.append(
                    partial(
                        self._contract_local,
                        ttno,
                        environment,
                        node,
                        node.tensor,
                        base_children,
                        inserted_from_parent[(operator_index, node)],
                        omit_child=output_child_index,
                        output="child",
                    )
                )
            for sibling_index, sibling in enumerate(node.children):
                if sibling_index == output_child_index:
                    continue
                messages = base_children.copy()
                messages[sibling_index] = inserted_to_parent[
                    (operator_index, sibling)
                ]
                calls.append(
                    partial(
                        self._contract_local,
                        ttno,
                        environment,
                        node,
                        node.tensor,
                        messages,
                        enode.environ_parent,
                        omit_child=output_child_index,
                        output="child",
                    )
                )
            return (operator_index, output_child), calls

        # Open the bra tensor at each node.  One term has the ket insertion at
        # the node itself; every other term imports a one-insertion edge
        # message.  This is the tree equivalent of the three diagrams in S10.
        def output_calls(operator_index, node):
            ttno = ttnos[operator_index]
            environment = environments[operator_index]
            node_index = self.reference.node_idx[node]
            enode = environment.node_list[node_index]
            base_children = list(enode.environ_children)
            calls = [
                partial(
                    self._contract_local,
                    ttno,
                    environment,
                    node,
                    vector.tensors[node_index],
                    base_children,
                    enode.environ_parent,
                    include_bra=False,
                    output="bra",
                )
            ]
            for child_index, child in enumerate(node.children):
                messages = base_children.copy()
                messages[child_index] = inserted_to_parent[(operator_index, child)]
                calls.append(
                    partial(
                        self._contract_local,
                        ttno,
                        environment,
                        node,
                        node.tensor,
                        messages,
                        enode.environ_parent,
                        include_bra=False,
                        output="bra",
                    )
                )
            if node.parent is not None:
                calls.append(
                    partial(
                        self._contract_local,
                        ttno,
                        environment,
                        node,
                        node.tensor,
                        base_children,
                        inserted_from_parent[(operator_index, node)],
                        include_bra=False,
                        output="bra",
                    )
                )
            return calls

        output_cotangents: Dict[object, np.ndarray] = {}
        messages_from_parent_seconds = 0.0
        local_projectors_seconds = 0.0
        if parallel_projectors:
            # Propagate the shared input insertion through the tree first.
            # Once every directed-edge message exists, P_v H_T psi for all
            # nodes v is independent and can be scheduled in one wide pool.
            parent_started = time.perf_counter()
            for level in self._preorder_levels:
                edges = [
                    (node, child_index, child)
                    for node in level
                    for child_index, child in enumerate(node.children)
                ]
                parent_groups = []
                for operator_index in range(len(ttnos)):
                    for edge in edges:
                        output_child, calls = calls_from_parent(
                            operator_index, edge
                        )
                        parent_groups.append((output_child, calls))
                evaluated = self._evaluate_contraction_groups(parent_groups)
                inserted_from_parent.update(evaluated)
            messages_from_parent_seconds = time.perf_counter() - parent_started

            projector_started = time.perf_counter()
            output_groups = [
                (
                    (operator_index, node),
                    output_calls(operator_index, node),
                )
                for operator_index in range(len(ttnos))
                for node in self.reference
            ]
            output_cotangents.update(
                self._evaluate_contraction_groups(output_groups)
            )
            local_projectors_seconds = time.perf_counter() - projector_started
        elif dependency_dag:
            # Release a node as soon as its own root-side insertion message is
            # ready.  This preserves every contraction and reduction order of
            # the level wavefront while removing unrelated depth barriers.
            def groups_at_node(operator_index, node):
                # Start terminal output work together with the dependency
                # path.  At high worker counts this overlap avoids leaving a
                # tail of local contractions after all edge messages finish.
                groups = [
                    (
                        ("output", operator_index, node),
                        output_calls(operator_index, node),
                    )
                ]
                for child_index, child in enumerate(node.children):
                    output_child, calls = calls_from_parent(
                        operator_index, (node, child_index, child)
                    )
                    groups.append((("parent", *output_child), calls))
                return groups

            def top_down_completed(owner, value):
                kind, operator_index, node = owner
                if kind == "output":
                    output_cotangents[(operator_index, node)] = value
                    return ()
                inserted_from_parent[(operator_index, node)] = value
                return groups_at_node(operator_index, node)

            root = next(node for node in self.reference if node.parent is None)
            self._evaluate_contraction_dag(
                (
                    group
                    for operator_index in range(len(ttnos))
                    for group in groups_at_node(operator_index, root)
                ),
                top_down_completed,
            )
        else:
            # Serial execution keeps the low-overhead level traversal.
            for level in self._preorder_levels:
                edges = [
                    (node, child_index, child)
                    for node in level
                    for child_index, child in enumerate(node.children)
                ]
                parent_groups = []
                for operator_index in range(len(ttnos)):
                    for edge in edges:
                        output_child, calls = calls_from_parent(
                            operator_index, edge
                        )
                        parent_groups.append((("parent", output_child), calls))
                output_groups = [
                    (
                        ("output", operator_index, node),
                        output_calls(operator_index, node),
                    )
                    for operator_index in range(len(ttnos))
                    for node in level
                ]
                evaluated = self._evaluate_contraction_groups(
                    parent_groups + output_groups
                )
                inserted_from_parent.update(
                    {
                        (operator_index, child): evaluated[
                            ("parent", (operator_index, child))
                        ]
                        for operator_index in range(len(ttnos))
                        for _node, _child_index, child in edges
                    }
                )
                output_cotangents.update(
                    {
                        (operator_index, node): evaluated[
                            ("output", operator_index, node)
                        ]
                        for operator_index in range(len(ttnos))
                        for node in level
                    }
                )
        top_down_output_seconds = (
            time.perf_counter() - started - to_parent_seconds
        )

        def finish_output(node):
            node_index = self.reference.node_idx[node]
            cotangent = _sum_arrays(
                output_cotangents[(operator_index, node)]
                for operator_index in range(len(ttnos))
            )
            parent_dim = node.tensor.shape[-1]
            cotangent_matrix = cotangent.reshape(-1, parent_dim)
            # raw[p,r] = sum_s result[p,s] metric[r,s]
            tangent_matrix = (
                cotangent_matrix @ self.parent_metric_pinv[node_index].T
            )
            tangent = tangent_matrix.reshape(node.tensor.shape)
            if energy_shift != 0.0:
                tangent = tangent - energy_shift * vector.tensors[node_index]
            return tangent

        metric_started = time.perf_counter()
        output_tensors = self._parallel_map(finish_output, self.reference)
        metric_seconds = time.perf_counter() - metric_started
        gauge_started = time.perf_counter()
        result = self.vector(output_tensors, project=True)
        gauge_seconds = time.perf_counter() - gauge_started
        self.synchronize()
        total_seconds = time.perf_counter() - started
        self.last_apply_timing_s = {
            "operator_components": len(ttnos),
            "parallel_projectors": parallel_projectors,
            "parallel_tangent_components": False,
            "tree_schedule": (
                "dependency_dag"
                if dependency_dag
                else (
                    "level_wavefront"
                    if self._executor is not None
                    else "serial_levels"
                )
            ),
            "messages_to_parent": to_parent_seconds,
            "messages_from_parent": messages_from_parent_seconds,
            "local_projectors": local_projectors_seconds,
            "top_down_and_local_contractions": top_down_output_seconds,
            "metric_application": metric_seconds,
            "gauge_projection": gauge_seconds,
            "total": total_seconds,
        }
        self.apply_timing_s["calls"] += 1
        for name, seconds in self.last_apply_timing_s.items():
            if name in self.apply_timing_s:
                self.apply_timing_s[name] += seconds
        return result

    def project_reference(self, ttno) -> TTNSTangentVector:
        r"""Project ``ttno |reference>`` into this tangent sector.

        Unlike :meth:`apply`, the operator may carry charge.  This is the
        tree-network counterpart of the charged excitation initialization in
        the paper's Eq. (S11).
        """

        TTNEnviron._assert_compatible_basis(self.reference.basis, ttno.basis)
        operator_qntot = np.asarray(ttno.qntot, dtype=int).reshape(-1)
        if operator_qntot.shape != self.reference_qntot.shape:
            raise ValueError("TTNO and reference quantum-number sizes differ")
        expected = self.reference_qntot + operator_qntot
        if np.any(expected != self.target_qntot):
            raise ValueError(
                "TTNO maps the reference to quantum number "
                f"{expected.tolist()}, not the tangent target "
                f"{self.target_qntot.tolist()}"
            )

        # Source projection is normally a one-off operation, so retaining its
        # (potentially large) charged-operator environment would waste memory.
        environment = TTNEnviron(self.reference, ttno)
        def project_node(item):
            node_index, node = item
            enode = environment.node_list[node_index]
            cotangent = self._contract_local(
                ttno,
                environment,
                node,
                node.tensor,
                list(enode.environ_children),
                enode.environ_parent,
                include_bra=False,
                output="bra",
            )
            parent_dim = node.tensor.shape[-1]
            cotangent_matrix = cotangent.reshape(-1, parent_dim)
            tangent_matrix = (
                cotangent_matrix @ self.parent_metric_pinv[node_index].T
            )
            return tangent_matrix.reshape(node.tensor.shape)

        tensors = self._parallel_map(
            project_node, enumerate(self.reference.node_list)
        )
        return self.vector(tensors, project=True)

    def to_dense(self, vector: TTNSTangentVector) -> np.ndarray:
        self._check_vector(vector)
        result = None
        for node_index, tangent in enumerate(vector.tensors):
            component = self.reference.copy()
            component.node_list[node_index].tensor = tangent.copy()
            dense = component.todense().reshape(-1)
            result = dense if result is None else result + dense
        if result is None:
            return np.array([], dtype=float)
        return result
