"""Amplitude-safe, bounded MPS algebra for Liouvillian eigensolvers.

Every iterative vector is an ordinary :class:`Mps`.  ``compress`` and
``combine`` enforce a retained bond cap; intermediate direct sums have at
most the sum of their input bond dimensions.  ``bond_cap=None`` is reserved
for residual audits: exact MPO images/direct sums may temporarily have larger
bonds, but are never converted to full Hilbert-space arrays.  QR gauges make
their small residual norms much more reliable than subtraction of large Gram
matrix entries.  Local QR/SVD tensors and scalar contraction environments
are the only arrays besides the eigensolver's small projected matrices.

Renormalizer stores site tensors on the host.  Contraction environments stay
on its selected NumPy/CuPy backend throughout each contraction; only the
resulting scalar is transferred back.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from opt_einsum import contract

from renormalizer.mps.backend import xp
from renormalizer.mps.matrix import asxp
from renormalizer.mps.mps import Mps
from renormalizer.utils import CompressConfig, CompressCriteria


def _check_state(state: Mps) -> None:
    if not isinstance(state, Mps) or state.site_num < 1:
        raise TypeError("Expected a nonempty native Renormalizer Mps")
    if not np.isfinite(complex(state.coeff)):
        raise FloatingPointError("MPS coefficient is not finite")


def _check_pair(left: Mps, right: Mps) -> None:
    _check_state(left)
    _check_state(right)
    if left.site_num != right.site_num or tuple(left.pbond_list) != tuple(right.pbond_list):
        raise ValueError("MPS site counts and physical dimensions must match")


def _explicit_zero(state: Mps) -> bool:
    return complex(state.coeff) == 0 or any(not core.array.any() for core in state)


def _zero_like(state: Mps) -> Mps:
    """A bond-one zero vector in the input sector; no QR of zero tensors."""
    result = state.copy()
    result.move_qnidx(0)
    result.qn = [labels[:1].copy() for labels in result.qn]
    for site, dimension in enumerate(result.pbond_list):
        result[site] = np.zeros((1, int(dimension), 1), dtype=result.dtype)
    result.coeff = 1.0
    result.to_right = True
    return result


def absorb_coefficient(state: Mps) -> Mps:
    """Copy an MPS and absorb its scalar coefficient into the center tensor."""
    _check_state(state)
    if _explicit_zero(state):
        return _zero_like(state)
    result = state.copy()
    coefficient = complex(result.coeff)
    if coefficient.imag:
        result.to_complex(inplace=True)
    else:
        coefficient = coefficient.real
    if coefficient != 1:
        center = int(result.qnidx)
        result[center] = result[center].array * coefficient
    result.coeff = 1.0
    return result


def _right_qr(state: Mps) -> Mps:
    """Right canonicalize an owned MPS, including exact-zero intermediates."""
    if _explicit_zero(state):
        return _zero_like(state)
    state.move_qnidx(state.site_num - 1)
    state.to_right = False
    # Native canonicalise asserts on an exactly zero local tensor.  This can
    # legitimately appear while gauging an exactly cancelling direct sum.
    for site in range(state.site_num - 1, 0, -1):
        if not state[site].array.any():
            return _zero_like(state)
        state._push_cano(site)
    if not state[0].array.any():
        return _zero_like(state)
    state.qnidx = 0
    state.to_right = True
    return state


def norm(state: Mps) -> float:
    """Euclidean/HS norm, including ``coeff``, evaluated after a QR sweep.

    The input is not changed.  This avoids cancellation in a Gram formula
    when ``state`` represents a residual formed by subtracting large MPSs.
    """
    owned = absorb_coefficient(state)
    if _explicit_zero(owned):
        return 0.0
    owned = _right_qr(owned)
    # A scaled Frobenius norm avoids unnecessary under/overflow in squaring.
    center = owned[0].array
    largest = float(np.max(np.abs(center)))
    if largest == 0:
        return 0.0
    value = largest * float(np.linalg.norm(center / largest))
    if not np.isfinite(value):
        raise FloatingPointError("MPS norm is not finite")
    return value


def inner(left: Mps, right: Mps) -> complex:
    """Return ``<left|right>`` with both scalar amplitudes included."""
    _check_pair(left, right)
    amplitude = complex(left.coeff).conjugate() * complex(right.coeff)
    if amplitude == 0:
        return 0.0j
    environment = xp.ones((1, 1), dtype=xp.complex128)
    for bra, ket in zip(left, right):
        environment = contract(
            "ab,apc,bpd->cd", environment, asxp(bra).conj(), asxp(ket),
            optimize="greedy",
        )
    return amplitude * complex(environment[0, 0])


def expectation(mpo, left: Mps, right: Mps) -> complex:
    """Return ``<left|mpo|right>`` through a direct transition contraction.

    The MPO is not applied to build a higher-rank MPS.  Small imaginary
    parts are preserved, unlike convenience routines that coerce nearly-real
    expectation values to floats.  No normalization is implicit.
    """
    _check_pair(left, right)
    if mpo.site_num != right.site_num:
        raise ValueError("MPO and MPS site counts must match")
    amplitude = complex(left.coeff).conjugate() * complex(right.coeff)
    if amplitude == 0:
        return 0.0j
    environment = xp.ones((1, 1, 1), dtype=xp.complex128)
    for bra, core, ket in zip(left, mpo, right):
        if core.shape[1:3] != (bra.shape[1], ket.shape[1]):
            raise ValueError("MPO and MPS physical dimensions must match")
        environment = contract(
            "awb,apc,wpqx,bqd->cxd", environment,
            asxp(bra).conj(), asxp(core), asxp(ket), optimize="greedy",
        )
    return amplitude * complex(environment[0, 0, 0])


def _validate_compression(bond_cap, cutoff: float) -> None:
    if bond_cap is not None and (
        isinstance(bond_cap, bool) or int(bond_cap) != bond_cap or bond_cap < 1
    ):
        raise ValueError("bond_cap must be a positive integer or None")
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("cutoff must be finite and strictly between zero and one")


def _compress_owned(state: Mps, bond_cap, cutoff: float) -> Mps:
    state = _right_qr(state)
    if _explicit_zero(state) or state.site_num == 1 or bond_cap is None:
        return state
    state.compress_config = CompressConfig(
        CompressCriteria.both, max_bonddim=int(bond_cap), threshold=float(cutoff)
    )
    state.compress(check_canonical=False)
    if max(state.bond_dims) > bond_cap:
        raise RuntimeError("Native MPS compression exceeded the requested bond cap")
    return state


def compress(state: Mps, bond_cap: int | None, cutoff: float = 1e-12) -> Mps:
    """Copy, absorb the amplitude, and SVD-compress to a strict bond cap.

    ``cutoff`` is the threshold on singular values normalized by their
    Euclidean norm, following ``CompressConfig``.  It is not a guaranteed
    global relative-error tolerance.  ``None`` disables SVD truncation and
    performs only exact QR gauge reduction, for temporary residual audits.
    """
    _validate_compression(bond_cap, cutoff)
    return _compress_owned(absorb_coefficient(state), bond_cap, cutoff)


def _add_owned(left: Mps, right: Mps) -> Mps:
    if left.site_num == 1:
        # MatrixProduct.add builds both boundary blocks, which is incorrect
        # when the first and last site are the same tensor.
        if right.is_complex and not left.is_complex:
            left.to_complex(inplace=True)
        left[0] = left[0].array + right[0].array
        return left
    # Addition concatenates bond QNs; align their left/right conventions.
    right.move_qnidx(left.qnidx)
    right.to_right = left.to_right
    return left.add(right)


def combine(
    states: Sequence[Mps], weights: Sequence[complex],
    bond_cap: int | None, cutoff: float = 1e-12,
) -> Mps:
    """Streaming sum ``sum(weight * state)`` with a cap after every add.

    Inputs and their scalar coefficients are never modified.  Compression is
    order dependent and nonlinear: certify the *returned* MPS against the
    original MPO.  With ``bond_cap=None``, the sum receives QR only and is
    suitable for residual audits without discarding small residual modes.
    """
    _validate_compression(bond_cap, cutoff)
    states, weights = list(states), list(weights)
    if not states or len(states) != len(weights):
        raise ValueError("Nonempty states and matching weights are required")
    result = None
    for state, weight in zip(states, weights):
        _check_pair(states[0], state)
        if not np.array_equal(states[0].qntot, state.qntot):
            raise ValueError("Cannot add MPSs from different total-QN sectors")
        weight = complex(weight)
        if not np.isfinite(weight):
            raise FloatingPointError("MPS combination weight is not finite")
        if weight == 0 or _explicit_zero(state):
            continue
        term = absorb_coefficient(state)
        if weight.imag and not term.is_complex:
            term.to_complex(inplace=True)
        term[term.qnidx] = term[term.qnidx].array * (weight if weight.imag else weight.real)
        term = _compress_owned(term, bond_cap, cutoff)
        if result is None or _explicit_zero(result):
            result = term
        else:
            result = _compress_owned(_add_owned(result, term), bond_cap, cutoff)
    return _zero_like(states[0]) if result is None else result


def eigen_residual(mpo, state: Mps, value: complex) -> float:
    """Return ``||mpo*state-value*state|| / ||state||`` for the original MPO.

    Only a temporary exact MPO image and direct-sum MPS are built.  Their
    bonds may exceed the iterative cap; no global vector, matrix or squared
    MPO is formed.  A zero candidate has no defined eigen-residual.
    """
    if not np.isfinite(complex(value)):
        raise ValueError("Eigenvalue must be finite")
    denominator = norm(state)
    if denominator == 0:
        raise ValueError("A zero MPS is not an eigenvector")
    image = mpo.apply(state)
    residual = combine([image, state], [1.0, -complex(value)], bond_cap=None)
    return norm(residual) / denominator


def product_trace(model) -> Mps:
    """Trace-bra MPS for an explicitly paired local operator basis.

    Calling this function asserts that each site's physical index is
    ``ket*d + bra`` with local operator dimension ``d**2``.  The function
    returns the unnormalized vectorized identity, whose overlap with a
    vectorized operator is its trace.  The MPO alone cannot establish this
    convention.  Every occupied diagonal state at each site must have the
    same local QN; sector-restricted or other vectorizations require a
    separately supplied trace MPS.
    """
    if model.nsite < 1:
        raise ValueError("A trace MPS needs a nonempty model")
    state = Mps()
    state.model = model
    state.build_empty_mp(model.nsite)
    state.qn = [np.zeros((1, model.qn_size), dtype=int)]
    for site, basis in enumerate(model.basis):
        dimension = math.isqrt(int(basis.nbas))
        if dimension * dimension != basis.nbas:
            raise ValueError(f"Site {site} dimension is not a paired square d**2")
        diagonal = np.arange(dimension, dtype=int) * (dimension + 1)
        diagonal_qn = np.asarray(basis.sigmaqn)[diagonal]
        if not np.all(diagonal_qn == diagonal_qn[0]):
            raise ValueError(
                f"Site {site} identity mixes QNs; supply an explicit trace MPS"
            )
        core = np.zeros((1, int(basis.nbas), 1), dtype=state.dtype)
        core[0, diagonal, 0] = 1.0
        state[site] = core
        state.qn.append(state.qn[-1] + diagonal_qn[0].reshape(1, model.qn_size))
    state.qntot = state.qn[-1][0].copy()
    state.qnidx = model.nsite
    state.move_qnidx(model.nsite - 1)
    state.to_right = False
    return state
