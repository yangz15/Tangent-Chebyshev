"""Complex, non-Hermitian reduced tangent Krylov pencils.

The trial and test frames V and W define A=W† K V, S=W† V,
b=W† source and d=measurement† V.  No Hermitian symmetrization is made.
The transfer function is d (z S-A)^(-1) b.  This small-matrix module is
independent of the tensor representation used to construct the pencil.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la


class ReducedTangentPencil:
    """A regular Petrov--Galerkin pencil with complex sources/measurements.

    ``source_projection`` is the dual right-hand side b, **not** the
    coordinate vector S^(-1)b. ``measurements`` contains already conjugated
    bra rows. A non-Hermitian S is allowed; singular/ill-conditioned metrics
    are rejected rather than silently changing the test/trial spaces.
    Arrays are copied to freeze the reduction. Dense operations scale with
    the Krylov dimension, never with the many-body Hilbert-space dimension.
    """

    def __init__(self, operator, overlap, source_projection, measurements,
                 metric_rcond=1e-12):
        if not np.isfinite(metric_rcond) or not 0 < metric_rcond < 1:
            raise ValueError("metric_rcond must lie strictly between zero and one")
        self.operator = np.array(operator, dtype=complex, copy=True)
        self.overlap = np.array(overlap, dtype=complex, copy=True)
        self.source_projection = np.array(source_projection, dtype=complex, copy=True)
        self.measurements = np.array(measurements, dtype=complex, copy=True)
        if self.source_projection.ndim == 1:
            self.source_projection = self.source_projection[:, None]
        if self.measurements.ndim == 1:
            self.measurements = self.measurements[None, :]
        a, s, b, d = (self.operator, self.overlap, self.source_projection,
                       self.measurements)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or s.shape != a.shape:
            raise ValueError("operator and overlap must be square matrices of equal size")
        self.dimension = len(a)
        if (b.ndim != 2 or b.shape[0] != len(a) or d.ndim != 2
                or d.shape[1] != len(a)):
            raise ValueError("source and measurement dimensions do not match the pencil")
        if not all(np.all(np.isfinite(v)) for v in (a, s, b, d)):
            raise ValueError("pencil, sources and measurements must be finite")
        singular = la.svdvals(s)
        self.metric_condition = (1. if not len(a) else
                                 float(singular[0]/singular[-1]) if singular[-1] else np.inf)
        if len(a) and (singular[-1] <= metric_rcond*singular[0]):
            raise ValueError("singular or ill-conditioned tangent overlap metric; "
                             "repair the basis or explicitly choose a rank reduction")
        self.metric_rcond = metric_rcond

    @classmethod
    def from_arnoldi(cls, result, dimension=None, **kwargs):
        """Use the contracted pencil; older coordinate Arnoldi results use S=I.

        A leading subspace uses the corresponding principal pencil, including
        the physical overlap with its Arnoldi tail already present in A.
        It must not be replaced by S times the leading Hessenberg block.
        """
        size = int(result["dimension"])
        m = size if dimension is None else dimension
        if not isinstance(m, (int, np.integer)) or not 0 <= m <= size:
            raise ValueError("dimension must be an integer between zero and the saved size")
        s = np.asarray(result.get("overlap", np.eye(size)), dtype=complex)
        a = np.asarray(result["projected_operator"] if "projected_operator" in result
                       else result["hessenberg"][:size, :size])
        if "source_projection" in result:
            b = np.asarray(result["source_projection"])
        else:
            c = np.zeros(size, complex)
            if size:
                c[0] = result["source_norm"]
            b = s @ c
        d = np.asarray(result["left_overlaps"])
        return cls(a[:m, :m], s[:m, :m], b[:m], d[..., :m], **kwargs)

    def coordinate_model(self):
        """Return S^(-1)A and S^(-1)b using solves, without an explicit inverse."""
        if not self.dimension:
            return self.operator.copy(), self.source_projection.copy()
        return (la.solve(self.overlap, self.operator, assume_a="gen"),
                la.solve(self.overlap, self.source_projection, assume_a="gen"))

    def resolvent(self, z):
        """d (z S-A)^(-1)b; output shape z.shape+(n_measurements,n_sources)."""
        points = np.asarray(z, dtype=complex)
        if not np.all(np.isfinite(points)):
            raise ValueError("resolvent points must be finite")
        out = np.zeros(points.shape + (len(self.measurements),
                                       self.source_projection.shape[1]), complex)
        if self.dimension:
            for index in np.ndindex(points.shape):
                out[index] = self.measurements @ la.solve(
                    points[index]*self.overlap-self.operator,
                    self.source_projection, assume_a="gen")
        return out

    def time_response(self, times, factor=1):
        """d exp(factor S^(-1)A t) S^(-1)b, including Jordan blocks.

        Use factor=-1j for Hamiltonian evolution and factor=1 for a generator.
        Matrix exponentials avoid a diagonalizability assumption.
        """
        times = np.asarray(times, dtype=complex)
        if not np.all(np.isfinite(times)) or not np.isfinite(factor):
            raise ValueError("times and evolution factor must be finite")
        out = np.zeros(times.shape + (len(self.measurements),
                                      self.source_projection.shape[1]), complex)
        if self.dimension:
            k, c = self.coordinate_model()
            for index in np.ndindex(times.shape):
                out[index] = self.measurements @ la.expm(factor*times[index]*k) @ c
        return out

    def eigenpairs(self):
        """Generalized left/right Ritz pairs and normwise backward residuals.

        SciPy left columns obey A†y=conj(omega) S†y. Pairs are normalized by
        y†Sx when it is safe, never by y†x or an absolute spectral weight.
        Defective pencils still have meaningful resolvents, but generally no
        complete simple-pole expansion; see :meth:`poles_weights`.
        """
        if not self.dimension:
            return dict(poles=np.empty(0, complex), left=self.operator.copy(),
                        right=self.operator.copy(), cross_overlap=self.operator.copy(),
                        right_residuals=np.empty(0), left_residuals=np.empty(0),
                        eigenvector_condition=1., metric_condition=1.)
        a, s = self.operator, self.overlap
        poles, left, right = la.eig(a, s, left=True, right=True)
        if not np.all(np.isfinite(poles)):
            raise la.LinAlgError("nonfinite Ritz eigenvalue in a regular tangent pencil")
        norms = np.einsum("ij,ij->j", left.conj(), s @ right)
        threshold = 64*np.finfo(float).eps*la.norm(s, 2)
        safe = abs(norms) > threshold
        left[:, safe] /= norms[safe].conj()[None, :]
        scale = la.norm(a, 2)+abs(poles)*la.norm(s, 2)
        scale = np.maximum(scale, np.finfo(float).tiny)
        rr = la.norm(a @ right-(s @ right)*poles, axis=0)/(scale*la.norm(right, axis=0))
        lr = la.norm(a.conj().T @ left-(s.conj().T @ left)*poles.conj(), axis=0)
        lr /= scale*la.norm(left, axis=0)
        return dict(poles=poles, left=left, right=right,
                    cross_overlap=left.conj().T @ s @ right,
                    right_residuals=rr, left_residuals=lr,
                    eigenvector_condition=float(np.linalg.cond(right)),
                    metric_condition=self.metric_condition)

    def poles_weights(self, condition_limit=1e10):
        """Return poles, complex residue tensors (obs, mode, source), and audit.

        The inverse eigenvector action handles degenerate eigenspaces without
        assuming independently returned left columns are mutually dual.
        An incomplete/ill-conditioned modal basis raises instead of producing
        misleading residues. Direct resolvent/time methods remain available.
        """
        if not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError("condition_limit must be finite and greater than one")
        pairs = self.eigenpairs()
        r = pairs["right"]
        if pairs["eigenvector_condition"] > condition_limit:
            raise la.LinAlgError("defective or ill-conditioned Ritz eigenbasis; "
                                "use direct resolvent/time_response")
        _, c = self.coordinate_model()
        amplitudes = la.solve(r, c) if self.dimension else c
        weights = (self.measurements @ r)[:, :, None]*amplitudes[None, :, :]
        audit = dict(eigenvector_condition=pairs["eigenvector_condition"],
                     metric_condition=pairs["metric_condition"],
                     maximum_right_residual=float(np.max(pairs["right_residuals"], initial=0)),
                     maximum_left_residual=float(np.max(pairs["left_residuals"], initial=0)),
                     weight_sum_error=float(la.norm(weights.sum(axis=1)-self.measurements @ c)))
        return pairs["poles"], weights, audit
