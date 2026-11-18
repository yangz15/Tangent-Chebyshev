"""Non-Abelian SU(2)-reduced one-site TTNS tangent vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from renormalizer.tn.tangent import TTNSTangentSpace, TTNSTangentVector
from renormalizer.utils.su2 import SU2Basis, su2_coupled_embedding


@dataclass
class SU2TTNSTangentVector:
    """A TTNS tangent vector stored in reduced fusion/degeneracy spaces."""

    space: "SU2TTNSTangentSpace"
    coefficients: List[np.ndarray]

    def copy(self):
        return type(self)(self.space, [array.copy() for array in self.coefficients])

    def dot(self, other) -> complex:
        self.space._check_vector(other)
        return sum(
            (
                np.vdot(left, right)
                for left, right in zip(self.coefficients, other.coefficients)
            ),
            0.0j,
        )

    @property
    def norm(self) -> float:
        value = self.dot(self).real
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        return float(np.sqrt(value))

    def scale(self, value, inplace: bool = False):
        result = self if inplace else self.copy()
        result.coefficients = [value * array for array in result.coefficients]
        return result

    def axpy(self, value, other):
        self.space._check_vector(other)
        self.coefficients = [
            array + value * increment
            for array, increment in zip(self.coefficients, other.coefficients)
        ]
        return self

    def normalize(self):
        norm = self.norm
        if norm == 0.0:
            raise ValueError("cannot normalize a zero SU(2) tangent vector")
        return self.scale(1.0 / norm, inplace=True)

    def to_dense(self) -> np.ndarray:
        return self.space.to_dense(self)

    def as_abelian(self) -> TTNSTangentVector:
        return self.space.as_abelian(self)


class SU2TTNSTangentSpace:
    r"""SU(2)-reduced subspace of a root-gauge TTNS tangent space.

    ``node_axes[v]`` follows the exact dense-axis order of TTNS node ``v``:
    child branches, local physical legs, and finally the root-side branch.
    This supports arbitrary tree coordination and fusion multiplicities.  The
    local reduced embeddings are orthonormalized in the non-trivial root-side
    metric used by :class:`TTNSTangentSpace`.
    """

    def __init__(
        self,
        reference,
        node_axes: Sequence[Sequence[SU2Basis]],
        target_two_j: int,
        target_two_m: Optional[int] = None,
        target_qntot=None,
        tolerance: float = 1.0e-10,
        metric_rcond: float = 1.0e-12,
    ):
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        self.tolerance = float(tolerance)
        self.base = TTNSTangentSpace(
            reference,
            target_qntot=target_qntot,
            metric_rcond=metric_rcond,
        )
        self.reference = self.base.reference
        self.target_qntot = self.base.target_qntot
        self.qntot = self.target_qntot
        self.target_two_j = int(target_two_j)
        if target_two_m is None:
            target_two_m = target_two_j
        self.target_two_m = int(target_two_m)
        self.node_axes = tuple(tuple(axes) for axes in node_axes)
        if len(self.node_axes) != self.base.node_num:
            raise ValueError("node_axes must contain one entry for each TTNS node")

        self.embeddings: List[np.ndarray] = []
        self.fusion_channels: List[Tuple] = []
        self.coordinate_embeddings: List[np.ndarray] = []
        for node_index, (axes, shape) in enumerate(
            zip(self.node_axes, self.base.shapes)
        ):
            if len(axes) != len(shape) or not all(
                isinstance(axis, SU2Basis) for axis in axes
            ):
                raise TypeError(
                    "each TTNS tensor axis must have one SU2Basis description"
                )
            axis_shape = tuple(axis.dimension for axis in axes)
            if axis_shape != shape:
                raise ValueError(
                    f"SU(2) axes at node {node_index} have shape {axis_shape}, "
                    f"but its tangent tensor has shape {shape}"
                )
            embedding, channels = su2_coupled_embedding(
                axes, self.target_two_j, self.target_two_m
            )
            self.fusion_channels.append(channels)
            self.embeddings.append(embedding)
            self.coordinate_embeddings.append(
                self._build_coordinate_embedding(node_index, embedding)
            )
        if not any(matrix.shape[1] for matrix in self.coordinate_embeddings):
            raise ValueError("the selected SU(2) TTNS tangent sector is empty")

    def _metric_apply(self, node_index, flattened):
        shape = self.base.shapes[node_index]
        parent_dim = shape[-1]
        matrices = flattened.reshape(-1, parent_dim, flattened.shape[-1])
        metric = self.base.parent_metrics[node_index]
        return np.einsum("rsc,ts->rtc", matrices, metric, optimize=True).reshape(
            flattened.shape
        )

    def _build_coordinate_embedding(self, node_index, embedding):
        mask = self.base.qn_masks[node_index].reshape(-1)
        selected = embedding.copy()
        selected[~mask, :] = 0
        removed = np.linalg.norm(embedding - selected)
        scale = max(1.0, float(np.linalg.norm(embedding)))
        if removed > self.tolerance * scale:
            raise ValueError(
                f"SU(2) sector at node {node_index} is incompatible with the "
                "selected commuting Abelian charge"
            )
        if selected.shape[1] == 0:
            return np.zeros((selected.shape[0], 0), dtype=complex)
        gram_error = np.linalg.norm(
            selected.conj().T @ selected - np.eye(selected.shape[1])
        )
        if gram_error > self.tolerance * max(1, selected.shape[1]):
            raise ValueError("SU(2) fusion columns are not orthonormal")

        shape = self.base.shapes[node_index]
        isometry = self.base.reference.node_list[node_index].tensor.reshape(
            -1, shape[-1]
        )
        gauge_mask = self.base.gauge_qn_masks[node_index]
        candidate = np.empty_like(selected, dtype=np.result_type(selected, isometry))
        for column in range(selected.shape[1]):
            matrix = selected[:, column].reshape(-1, shape[-1])
            overlap = isometry.conj().T @ matrix
            overlap[~gauge_mask] = 0
            projected = matrix - isometry @ overlap
            support = self.base.parent_metric_support[node_index]
            if support is not None:
                projected = projected @ support.T
            dense = projected.reshape(-1)
            dense[~mask] = 0
            candidate[:, column] = dense

        outside = candidate - selected @ (selected.conj().T @ candidate)
        outside_norm = np.linalg.norm(outside)
        candidate_scale = max(1.0, float(np.linalg.norm(candidate)))
        if outside_norm > self.tolerance * candidate_scale:
            raise ValueError(
                f"TTNS canonical gauge at node {node_index} is not SU(2)-adapted; "
                f"relative intertwiner residual {outside_norm / candidate_scale:.3e}"
            )

        weighted = self._metric_apply(node_index, candidate)
        gram = candidate.conj().T @ weighted
        gram = 0.5 * (gram + gram.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        if not eigenvalues.size:
            return np.zeros((selected.shape[0], 0), dtype=candidate.dtype)
        cutoff = self.tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
        retained = eigenvalues > cutoff
        if not np.any(retained):
            return np.zeros((selected.shape[0], 0), dtype=candidate.dtype)
        return candidate @ (
            eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
        )

    @property
    def node_num(self):
        return self.base.node_num

    @property
    def dimension(self):
        return int(sum(matrix.shape[1] for matrix in self.coordinate_embeddings))

    @property
    def non_abelian(self):
        return True

    @property
    def reduced(self):
        return True

    def _check_vector(self, vector):
        if not isinstance(vector, SU2TTNSTangentVector) or vector.space is not self:
            raise ValueError("vectors belong to different SU(2) TTNS tangent spaces")
        if len(vector.coefficients) != self.node_num:
            raise ValueError("wrong number of reduced TTNS tangent tensors")
        for node_index, (coefficients, embedding) in enumerate(
            zip(vector.coefficients, self.coordinate_embeddings)
        ):
            if np.asarray(coefficients).shape != (embedding.shape[1],):
                raise ValueError(f"wrong reduced-coordinate shape at node {node_index}")

    def vector(self, coefficients: Iterable[np.ndarray]):
        coefficients = [np.asarray(array).copy() for array in coefficients]
        vector = SU2TTNSTangentVector(self, coefficients)
        self._check_vector(vector)
        return vector

    def zeros(self, dtype=complex):
        return self.vector(
            [np.zeros(matrix.shape[1], dtype=dtype) for matrix in self.coordinate_embeddings]
        )

    def random(self, seed=None, complex_values=True):
        rng = np.random.default_rng(seed)
        coefficients = []
        for matrix in self.coordinate_embeddings:
            array = rng.standard_normal(matrix.shape[1])
            if complex_values:
                array = array + 1j * rng.standard_normal(matrix.shape[1])
            coefficients.append(array)
        return self.vector(coefficients).normalize()

    def as_abelian(self, vector) -> TTNSTangentVector:
        self._check_vector(vector)
        tensors = [
            (embedding @ coefficients).reshape(shape)
            for embedding, coefficients, shape in zip(
                self.coordinate_embeddings,
                vector.coefficients,
                self.base.shapes,
            )
        ]
        return self.base.vector(tensors, project=False)

    def _from_abelian(self, vector, check=True):
        self.base._check_vector(vector)
        coefficients = []
        residual_squared = 0.0
        norm_squared = 0.0
        for node_index, (tensor, embedding) in enumerate(
            zip(vector.tensors, self.coordinate_embeddings)
        ):
            dense = tensor.reshape(-1, 1)
            weighted = self._metric_apply(node_index, dense)[:, 0]
            reduced = embedding.conj().T @ weighted
            reconstructed = embedding @ reduced
            residual = dense[:, 0] - reconstructed
            weighted_residual = self._metric_apply(
                node_index, residual.reshape(-1, 1)
            )[:, 0]
            residual_squared += max(
                0.0, float(np.vdot(residual, weighted_residual).real)
            )
            norm_squared += max(0.0, float(np.vdot(dense[:, 0], weighted).real))
            coefficients.append(reduced)
        residual_norm = np.sqrt(residual_squared)
        scale = max(1.0, np.sqrt(norm_squared))
        if check and residual_norm > self.tolerance * scale:
            raise ValueError(
                "TTNO action leaves the selected SU(2) irrep; the operator is "
                f"not an SU(2) scalar (relative residual {residual_norm / scale:.3e})"
            )
        return self.vector(coefficients)

    def apply(self, ttno, vector, energy_shift=0.0):
        dense_result = self.base.apply(
            ttno, self.as_abelian(vector), energy_shift=energy_shift
        )
        return self._from_abelian(dense_result, check=True)

    def project_reference(self, ttno):
        return self._from_abelian(self.base.project_reference(ttno), check=True)

    def to_dense(self, vector):
        return self.as_abelian(vector).to_dense()

    def gauge_residual(self, vector):
        return self.base.gauge_residual(self.as_abelian(vector))

    def su2_residual(self, vector):
        self._check_vector(vector)
        residual = 0.0
        for coefficients, coordinate, coupled in zip(
            vector.coefficients, self.coordinate_embeddings, self.embeddings
        ):
            tensor = coordinate @ coefficients
            residual = max(
                residual,
                float(np.linalg.norm(tensor - coupled @ (coupled.conj().T @ tensor))),
            )
        return residual
