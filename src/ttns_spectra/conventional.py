"""Whole-target variational coCheb with exact cached S/H contractions."""
import numpy as np
import cupy as cp
from renormalizer.tn import TTNO
from ._backend.gpu_charged import device_operator, variational_sum
from .problems import source, fit, Timer, UNIT
from .projected import CachedTreeBasis, flatten, flatten_operator
from .spectral import canonical_model, broaden_poles
from renormalizer.utils import constant


def shifted_operator(problem, center=0., width=1.):
    """Build (H-center*I)/width as an exact TTNO; no H|state> truncation."""
    if width <= 0:
        raise ValueError('width must be positive')
    return device_operator(TTNO(problem.tree,
        [t/width for t in problem.h.terms]+[problem.tree.identity_op*(-center/width)]))


def cocheb(problem, axis, *, case, vectors, grid, broadening,
           cutoff=None, fit_tolerance=1e-10, max_sweeps=200, cache_bytes=4*1024**3):
    """Compute conventional coCheb from a prepared problem.

    Each new state fits the whole target H't0 or 2H'tn-t(n-1), with the
    ground-state constraint for IR. Actual S_ij=<ti|tj> and H_ij=<ti|H'|tj>
    are measured once using cached, shape-grouped bra batches. Canonical
    overlap filtering precedes one reduced Hermitian solve. Moment doubling
    and Burg continuation are deliberately absent from conventional coCheb.

    Returns spectrum, raw moments, projected matrices, poles/weights, fit and
    metric diagnostics, and synchronized response-stage timings. Immutable
    host vectors are retained only during this call; no vector dump is needed.
    """
    if case not in ('IR','fast','fast_slow') or not isinstance(vectors,(int,np.integer)) or vectors < 2:
        raise ValueError('Unknown case or invalid vector count')
    grid=np.asarray(grid,dtype=float)
    if grid.ndim!=1 or len(grid)<2 or not np.isfinite(grid).all() or np.any(np.diff(grid)<=0):
        raise ValueError('grid must be finite and strictly increasing')
    if not np.isfinite(broadening) or broadening<=0 or fit_tolerance<=0 or max_sweeps<2 or cache_bytes<0:
        raise ValueError('Invalid broadening, fit tolerance, sweep limit or cache budget')
    if cutoff is not None and not 0<cutoff<1:
        raise ValueError('Relative overlap cutoff must lie strictly between zero and one')
    current, source_info = source(problem, axis, case)
    previous = None
    weight = source_info['weight']
    identity = device_operator(problem.identity)
    low, high = ((problem.center_h-problem.width, problem.center_h+problem.width)
                 if case=='IR' else problem.preparation['bounds'])
    center = (low+high)/2
    width = (high-low)/2 if case=='IR' else (high-low)/(2*.99)
    unit = constant.cm2au if case=='IR' else UNIT
    operator = shifted_operator(problem, center, width)
    flatop = flatten_operator(operator)
    basis = CachedTreeBasis(problem.ground, batch=64 if case=='IR' else 8, cache_bytes=cache_bytes)
    s = cp.zeros((vectors,vectors),complex)
    h = cp.zeros_like(s)
    stored, fits = [], []
    timer = Timer()
    for n in range(vectors):
        if n:
            with timer.stage('generate_compress_vectors'):
                terms = [(1. if n==1 else 2., operator, current)]
                if n>1:
                    terms.append((-1.,identity,previous))
                guess = variational_sum(current, terms, sweeps=1)
                following, info = fit(guess, terms,
                    constraint=problem.ground if case=='IR' else None, identity=identity,
                    tolerance=fit_tolerance, max_sweeps=max_sweeps)
                previous, current = current, following
                fits.append(info)
        with timer.stage('actual_SH_contractions'):
            ket = flatten(current)
            stored.append([cp.asnumpy(t) for t in ket])
            sc, hc = basis.column_gpu(stored,ket,flatop)
            if max(abs(complex(sc[-1]).imag),abs(complex(hc[-1]).imag)) > 1e-8*max(weight,1e-300):
                raise FloatingPointError('Non-Hermitian S/H diagonal')
            if not bool(cp.isfinite(sc).all()) or not bool(cp.isfinite(hc).all()) or float(sc[-1].real)>100*weight:
                raise FloatingPointError(f'coCheb norm/finite guard failed at vector {n}')
            s[:n+1,n],s[n,:n+1] = sc,sc.conj()
            h[:n+1,n],h[n,:n+1] = hc,hc.conj()
    cutoff = (1e-12 if case=='IR' else 1e-10) if cutoff is None else cutoff
    with timer.stage('canonical_reduction'):
        poles,weights,heff,b,diagnostic = canonical_model(s/weight,h/weight,weight,cutoff)
    energies = (center-problem.energy+width*poles)/unit
    def broaden():
        return broaden_poles(energies,weights,grid,broadening,lorentz=case!='IR')
    broaden()  # Exclude first kernel compilation, consistent with publication timing.
    with timer.stage('final_spectrum'):
        curve = broaden()
    return dict(spectrum=curve, grid=np.asarray(grid), raw_moments=cp.asnumpy(s[0]),
        overlap=cp.asnumpy(s), projected_hamiltonian_scaled=cp.asnumpy(h),
        poles=cp.asnumpy(energies), weights=cp.asnumpy(weights), source_weight=weight,
        center=center, width=width, fits=fits, reduction=diagnostic,
        computed_coefficients=vectors, recurrence_updates=vectors-1,
        stage_seconds={k:v/1e9 for k,v in timer.ns.items()})
