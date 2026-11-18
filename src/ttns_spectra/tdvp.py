"""One-site TDVP comparison at the selected publication time grids."""
import numpy as np
import cupy as cp
from renormalizer.utils import constant
from .problems import source, Timer, UNIT
from .conventional import shifted_operator
from ._backend.gpu_charged import device_operator, matrix_element
from ._backend.tdvp_one_site import tdvp_one_site


def transform_correlation(case, times, correlation, grid, broadening):
    """Trapezoidal damped Fourier integral with Gaussian/Lorentz broadening."""
    unit = constant.cm2au if case=='IR' else UNIT
    t, corr = cp.asarray(times), cp.asarray(correlation)
    dt = cp.diff(t)
    quadrature = cp.zeros_like(t)
    quadrature[:-1] += dt/2
    quadrature[1:] += dt/2
    damping = cp.exp(-.5*(broadening*unit*t)**2) if case=='IR' else cp.exp(-broadening*unit*t)
    integrand = corr*damping*quadrature
    values = cp.empty(len(grid))
    for start in range(0,len(grid),128):
        values[start:start+128] = (cp.exp(1j*cp.asarray(grid[start:start+128,None])*unit*t)@integrand).real*unit/np.pi
    return cp.asnumpy(values)


def tdvp_response(problem, axis, *, case, dt, steps, grid, broadening):
    """Fresh symmetric one-site propagation; dt is atomic/scaled for IR/fast."""
    if case not in ('IR','fast') or not np.isfinite(dt) or dt<=0:
        raise ValueError('Require case IR/fast and a finite positive time step')
    if not isinstance(steps,(int,np.integer)) or steps<1 or broadening<=0:
        raise ValueError('Require positive integer steps and positive broadening')
    grid=np.asarray(grid,dtype=float)
    initial, info = source(problem,axis,case)
    state = initial.copy().to_complex()
    op = shifted_operator(problem,problem.energy)
    identity = device_operator(problem.identity)
    step_au = dt if case=='IR' else dt/UNIT
    times = [0.]
    correlation = [complex(matrix_element(initial,identity,state))]
    timer = Timer()
    for step in range(steps):
        with timer.stage('propagation_and_correlation'):
            tdvp_one_site(state,op,step_au,krylov_tol=1e-10,krylov_bytes=128*1024**2)
            times.append(times[-1]+step_au)
            correlation.append(complex(matrix_element(initial,identity,state)))
    times,correlation = np.asarray(times),np.asarray(correlation)
    transform_correlation(case,times,correlation,grid,broadening)
    with timer.stage('final_spectrum'):
        curve = transform_correlation(case,times,correlation,grid,broadening)
    norm = float(matrix_element(state,identity,state).real)
    drift = abs(norm/correlation[0].real-1)
    if drift > 1e-6:
        raise FloatingPointError(f'TDVP final norm drift: {drift}')
    return dict(grid=grid,spectrum=curve,times=times,correlation=correlation,
        final_norm_relative_drift=drift,steps=steps,dt=dt,
        stage_seconds={k:v/1e9 for k,v in timer.ns.items()})
