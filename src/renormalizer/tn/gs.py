import logging
from typing import List, Union, Tuple


import scipy

from renormalizer.lib import davidson
from renormalizer.mps.backend import (
    IMPORT_PRIMME_EXCEPTION,
    USE_GPU,
    np,
    primme,
    xp,
)
from renormalizer.mps.matrix import asnumpy, asxp
from renormalizer.tn.node import TreeNodeTensor
from renormalizer.tn.tree import TTNS, TTNO, TTNEnviron
from renormalizer.tn.hop_expr import hop_expr2


logger = logging.getLogger(__name__)


def optimize_ttns(ttns: TTNS, ttno: Union[TTNO, List[TTNO], Tuple[TTNO, ...]], procedure=None):
    """Optimize against one TTNO or an exact sum of compact TTNO chunks."""
    if procedure is None:
        procedure = ttns.optimize_config.procedure
    if isinstance(ttno, TTNO):
        ttnos = (ttno,)
    else:
        ttnos = tuple(ttno)
        if not ttnos or not all(isinstance(item, TTNO) for item in ttnos):
            raise TypeError("ttno must be a TTNO or a non-empty sequence of TTNOs")
    ttnes = tuple(TTNEnviron(ttns, item) for item in ttnos)
    e_list = []
    for m, percent in procedure:
        # todo: better converge condition
        micro_e = optimize_recursion(ttns.root, ttns, ttnos, ttnes, m, percent)
        logger.info(f"Micro e: {micro_e}")
        e_list.append(micro_e[-1])
    return e_list


def optimize_recursion(
    snode: TreeNodeTensor, ttns: TTNS, ttnos: Tuple[TTNO, ...],
    ttnes: Tuple[TTNEnviron, ...], m: Union[int, List[int]], percent: float = 0
) -> List[float]:
    """Optimize snode and all of its children"""
    assert snode.children  # 2 site can't do only one node
    micro_e = []
    for ichild, child in enumerate(snode.children):
        if child.children:
            # optimize snode + child
            e, c = optimize_2site(child, ttns, ttnos, ttnes)
            micro_e.append(e)
            # cano to child
            ttns.update_2site(child, c, m, percent, cano_parent=False)
            # update env
            for ttno, ttne in zip(ttnos, ttnes):
                ttne.update_2site(child, ttns, ttno)
            # recursive optimization
            micro_e_child = optimize_recursion(child, ttns, ttnos, ttnes, m)
            micro_e.extend(micro_e_child)

        # optimize snode + child
        e, c = optimize_2site(child, ttns, ttnos, ttnes)
        micro_e.append(e)
        # cano to snode
        ttns.update_2site(child, c, m, percent, cano_parent=True)
        # update env
        for ttno, ttne in zip(ttnos, ttnes):
            ttne.update_2site(child, ttns, ttno)
    return micro_e


def optimize_2site(
    snode: TreeNodeTensor, ttns: TTNS, ttnos: Tuple[TTNO, ...],
    ttnes: Tuple[TTNEnviron, ...]
):
    cguess = ttns.merge_with_parent(snode)
    qn_mask = ttns.get_qnmask(snode, include_parent=True)
    cguess = cguess[qn_mask].ravel()
    expressions = [
        hop_expr2(snode, ttns, ttno, ttne)
        for ttno, ttne in zip(ttnos, ttnes)
    ]
    hdiag = sum(item[1] for item in expressions)[qn_mask].ravel()

    assert ttns.optimize_config.nroots == 1
    algo: str = ttns.optimize_config.algo
    if USE_GPU and algo == "davidson":
        qn_mask_device = xp.asarray(qn_mask)

        def hop_device(x):
            cstruct = xp.zeros(qn_mask.shape, dtype=x.dtype)
            cstruct[qn_mask_device] = x
            return sum(expr(cstruct) for expr, _ in expressions)[
                qn_mask_device
            ].ravel()

        e, c, info = davidson_gpu_lowest(
            hop_device,
            hdiag,
            xp.asarray(cguess),
            tol=1.0e-12,
            max_cycle=getattr(ttns.optimize_config, 'gpu_max_cycle', 100),
            max_space=getattr(ttns.optimize_config, 'gpu_max_space', 12),
            residual_tol=getattr(ttns.optimize_config, 'gpu_residual_tol', None),
            acceptance_tol=getattr(ttns.optimize_config, 'gpu_acceptance_tol', None),
            return_info=True,
        )
        records = getattr(ttns.optimize_config, 'gpu_solver_records', None)
        if records is not None:
            records.append(info)
        return e, vec2tensor(asnumpy(c), qn_mask)

    def hop(x):
        cstruct = vec2tensor(x, qn_mask)
        ret = sum(expr(asxp(cstruct)) for expr, _ in expressions)[qn_mask].ravel()
        return asnumpy(ret)

    e, c = eigh_iterative(hop, hdiag, cguess, algo)
    c = vec2tensor(c, qn_mask)
    return e, c


def davidson_gpu_lowest(
    hop,
    hdiag,
    cguess,
    *,
    tol: float = 1.0e-12,
    max_cycle: int = 100,
    max_space: int = 12,
    residual_tol: float = None,
    acceptance_tol: float = None,
    return_info: bool = False,
):
    """Single-root Hermitian Davidson iteration resident on one CUDA device.

    Only the converged eigenvalue and tensor cross the PCIe boundary. The
    effective-Hamiltonian expression, trial vectors, operator images,
    residuals, and diagonal preconditioner remain in CuPy throughout each
    two-site micro-optimization.

    ``acceptance_tol`` optionally permits a finite approximate local solve
    after the iteration limit. Outer sweeps must independently check their
    final residual. Without it, an explicit residual_tol remains strict.
    """

    if not USE_GPU:
        raise RuntimeError("davidson_gpu_lowest requires the CuPy backend")
    hdiag = xp.asarray(hdiag).reshape(-1)
    trial = xp.asarray(cguess).reshape(-1)
    trial_norm = xp.linalg.norm(trial)
    if not np.isfinite(float(trial_norm.get())) or float(trial_norm.get()) == 0.0:
        raise ValueError("GPU Davidson received a zero or nonfinite initial guess")
    if max_cycle < 1 or max_space < 2:
        raise ValueError('Davidson requires max_cycle >= 1 and max_space >= 2')
    trial = trial / trial_norm
    vectors = [trial]
    images = []
    last_energy = None
    residual_tolerance = float(np.sqrt(tol) if residual_tol is None else residual_tol)
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError('Davidson residual tolerance must be positive')
    if acceptance_tol is not None and (
        not np.isfinite(acceptance_tol) or acceptance_tol < residual_tolerance
    ):
        raise ValueError('Davidson acceptance tolerance must be finite and >= residual tolerance')
    energy = None
    ritz = None
    best_energy = float('inf')
    best_ritz = None

    for _cycle in range(max_cycle):
        while len(images) < len(vectors):
            images.append(hop(vectors[len(images)]))
        basis = xp.stack(vectors, axis=0)
        action = xp.stack(images, axis=0)
        effective = basis.conj() @ action.T
        effective = 0.5 * (effective + effective.conj().T)
        eigenvalues, eigenvectors = xp.linalg.eigh(effective)
        energy_device = eigenvalues[0]
        coefficients = eigenvectors[:, 0]
        ritz = coefficients @ basis
        action_ritz = coefficients @ action
        residual = action_ritz - energy_device * ritz
        residual_norm = float(xp.linalg.norm(residual).get())
        energy = float(energy_device.real.get())
        if not np.isfinite(energy) or not np.isfinite(residual_norm):
            raise FloatingPointError('Nonfinite GPU Davidson energy or residual')
        if energy < best_energy:
            best_energy, best_ritz = energy, ritz.copy()
        if (
            last_energy is not None
            and abs(energy - last_energy) < tol
            and residual_norm < residual_tolerance
        ):
            best_energy, best_ritz = energy, ritz.copy()
            break
        last_energy = energy

        denominator = hdiag - energy_device + 1.0e-4
        denominator = xp.where(xp.abs(denominator) < 1.0e-8,
                               xp.where(denominator < 0, -1.0e-8, 1.0e-8), denominator)
        correction = residual / denominator
        for _pass in range(2):
            overlaps = basis.conj() @ correction
            correction = correction - overlaps @ basis
        correction_norm = float(xp.linalg.norm(correction).get())
        if correction_norm <= (1.0e-14 if residual_tol is not None else 1.0e-7):
            break
        correction = correction / correction_norm

        if len(vectors) >= max_space:
            # Preserve the new correction across restart instead of discarding it.
            vectors = [ritz / xp.linalg.norm(ritz), correction]
            images = []
        else:
            vectors.append(correction)

    if energy is None or ritz is None:
        raise RuntimeError("GPU Davidson failed before forming a Ritz vector")
    # Verify the returned state with a fresh action, independently of the
    # small projected eigensystem. The lowest-energy Ritz state is retained.
    ritz = best_ritz / xp.linalg.norm(best_ritz)
    action_ritz = hop(ritz)
    energy = float(xp.vdot(ritz, action_ritz).real.get())
    residual_norm = float(xp.linalg.norm(action_ritz - energy * ritz).get())
    if not np.isfinite(energy) or not np.isfinite(residual_norm):
        raise FloatingPointError('Nonfinite final GPU Davidson energy or residual')
    hard_tolerance = residual_tolerance if acceptance_tol is None else acceptance_tol
    if (residual_tol is not None or acceptance_tol is not None) and residual_norm > hard_tolerance:
        raise RuntimeError(
            f'GPU Davidson residual {residual_norm:.3e} exceeds acceptance limit '
            f'{hard_tolerance:.3e} after {_cycle+1} iterations'
        )
    approximate = residual_norm > residual_tolerance
    if approximate:
        logger.warning('GPU Davidson returns approximate local state: residual %.3e, '
                       'target %.3e, acceptance %.3e, iterations %d; outer convergence required',
                       residual_norm, residual_tolerance, hard_tolerance, _cycle + 1)
    if return_info:
        return energy, ritz, dict(iterations=_cycle + 1, residual=residual_norm,
                                 target=residual_tolerance, acceptance=hard_tolerance,
                                 approximate=approximate)
    return energy, ritz


def eigh_iterative(hop, hdiag, cguess, algo):
    hdiag = asnumpy(hdiag)
    cguess = asnumpy(cguess)
    h_dim = len(hdiag)

    if algo == "davidson":
        precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

        e, c = davidson(hop, cguess, precond, max_cycle=100, nroots=1, max_memory=64000, verbose=0)
    elif algo == "primme":
        if primme is None:
            logger.error("can not import primme")
            raise IMPORT_PRIMME_EXCEPTION
        precond = lambda x: scipy.sparse.diags(1 / (hdiag + 1e-4)) @ x
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop, matmat=hop)
        M = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=precond, matmat=hop)
        e, c = primme.eigsh(
            A,
            k=1,
            which="SA",
            v0=np.array(cguess).reshape(-1, 1),
            OPinv=M,
            method="PRIMME_DYNAMIC",
            tol=1e-6,
        )
        c = c[:, 0]
        e = e[0]
    elif algo == "arpack":
        A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop)
        e, c = scipy.sparse.linalg.eigsh(A, k=1, which="SA", v0=cguess)
        e = e[0]
    elif algo == "direct":
        # direct algorithm. Poor performance, debugging only.
        a_list = []
        for i in range(h_dim):
            a = np.zeros(h_dim)
            a[i] = 1
            a_list.append(hop(a))
        a = np.array(a_list)
        assert np.allclose(a, a.conj().T)
        evals, evecs = np.linalg.eigh(a)
        e = evals[0]
        c = evecs[:, 0]
    else:
        assert False

    return e, c


def vec2tensor(c, qn_mask):
    cstruct = np.zeros(qn_mask.shape, dtype=c.dtype)
    np.place(cstruct, qn_mask, c)
    return cstruct
