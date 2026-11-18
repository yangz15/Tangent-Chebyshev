"""The two notebook entry points, with explicit fresh/replay provenance."""
import csv
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from .paths import DATA

METHODS = ('tangent1','cocheb','tdvp1')


def publication_settings(case):
    """Return a fresh dictionary of the exact settings behind the target PDFs."""
    if case not in ('IR','fast'):
        raise ValueError('case must be IR or fast')
    return json.loads((DATA/'settings.json').read_text())[case]


def load_reference(case, bond=None, directory=None):
    """Build operators and load a checksum-verified prepared binary TTNS.

    Built-in references are IR D16 and fast D32. A directory made by
    `prepare_reference` may be passed for a different prepared reference.
    """
    from .problems import prepared, build, load, sha
    bond = publication_settings(case)['bond'] if bond is None else bond
    if directory is None:
        return prepared(case,bond)
    directory = Path(directory)
    record = json.loads((directory/'summary.json').read_text())
    if record['status']!='completed' or record['case']!=case or record['bond']!=bond:
        raise ValueError('Reference metadata does not match the requested problem')
    if sha(directory/'reference.npz')!=record['source_reference_sha256']:
        raise ValueError('Reference checksum mismatch')
    p=build(case,bond)
    p.ground=load(p.tree,directory/'reference.npz',bond)
    p.energy=record['energy']
    p.metadata,p.preparation=record['metadata'],record
    return p


def prepare_reference(case, bond, output):
    """Run binary DMRG (IR) or beta/2 purified imaginary time (aggregate).

    Writes reference.npz and summary.json to `output`. This expensive setup
    is excluded from the published response timings and from notebook defaults.
    """
    from .problems import prepare
    if case not in ('IR','fast','fast_slow') or bond < 1:
        raise ValueError('Invalid case or bond dimension')
    return prepare(case,bond,Path(output))


def run_tangent(problem, axis, *, case, actions, two_site=False, bounds=None):
    """Run one-/two-site TaChe and Burg reconstruction for a prepared model.

    Uses a verified cached interval for a bundled reference, or estimates the
    endpoints when no matching cache exists. The resulting dictionary includes
    computed moments, spectrum, actual action count, and response-only timings.
    """
    import cupy as cp
    from renormalizer.utils import constant
    from .problems import STUDY, UNIT, stamp
    from .chebyshev import chebyshev_moments, estimate_tangent_bounds
    from .tangent import TreeTangent
    from ._backend.gpu_charged import device_operator
    from .maxent import reconstruct_spectrum
    if problem.preparation['case']!=case:
        raise ValueError('The prepared reference does not match the requested case')
    config = STUDY[case]
    target = [0] if case=='IR' else [0,1]
    unit = constant.cm2au if case=='IR' else UNIT
    space = TreeTangent(problem.ground,target,two_site=two_site,remove_reference=case=='IR')
    h = device_operator(problem.h)
    source = space.project(device_operator(problem.dipoles[axis]),problem.ground)
    cached = False
    if bounds is None:
        path=DATA/f'results/{case}/D{problem.bond}/tangent{2 if two_site else 1}/shared_bounds.json'
        if path.exists():
            record=json.loads(path.read_text())
            f=record['fingerprint']
            if (f['reference_sha256']==problem.preparation['source_reference_sha256']
                    and f['two_site']==two_site and f['target']==target
                    and f['remove_reference']==(case=='IR')):
                bounds=record['bounds']
                cached=True
        if bounds is None:
            bounds=estimate_tangent_bounds(space,h,energy_shift=problem.energy,unit=unit)
    result=chebyshev_moments(space,h,source,bounds,actions)
    grid=np.linspace(*config['grid'])
    center=(result['center']-problem.energy)/unit
    width=result['width']/unit
    def reconstruct():
        return reconstruct_spectrum(result['moments'],center,width,grid,
            kernel='gaussian' if case=='IR' else 'lorentzian',
            broadening=config['sigma_cm'] if case=='IR' else config['eta_scaled'],
            loading=config['maximum_entropy_loading'])
    reconstruct()
    began=stamp()
    curve,info=reconstruct()
    result['stage_seconds']['final_spectrum']=(stamp()-began)/1e9
    result.update(grid=grid,spectrum=curve,reconstruction=info,bounds=list(bounds),bounds_cached=cached)
    return result


def _jsonable_result(result):
    """Keep diagnostics and timings in JSON; arrays are saved separately."""
    return {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}


def reproduce(case, *, mode='fresh', output='outputs', methods=METHODS, save_matrices=False):
    """Reproduce one publication panel from fresh responses or saved arrays.

    Fresh mode regenerates all selected TaChe/coCheb/TDVP responses from bundled
    ground/thermal references. Replay mode is CPU-only and explicitly reuses
    archived curves and timings. A fresh comparison is checked against those
    saved spectra, with measured differences reported rather than hidden.
    Returns a JSON-serializable report and writes PDFs, CSVs, and diagnostics.
    Set save_matrices=True to export the large coCheb S/H matrices as well.
    """
    if mode not in ('fresh','replay'):
        raise ValueError('mode must be fresh or replay')
    if tuple(methods)!=METHODS:
        raise ValueError('The publication figure requires tangent1, cocheb and tdvp1')
    spec=publication_settings(case)
    output=Path(output)/case.lower()
    output.mkdir(parents=True,exist_ok=True)
    source=np.genfromtxt(DATA/'published'/f'{case.lower()}_source_spectra.csv',delimiter=',',names=True)
    grid,reference=source['frequency'],source['reference']
    oldrows=[r for r in csv.DictReader((DATA/'published/source_timing.csv').open()) if r['case']==case]
    oldtimes={r['method']:float(r['seconds']) for r in oldrows}
    curves,rows={},[]
    started=time.time()
    metadata=dict(python=platform.python_version(),numpy=np.__version__,slurm_job=os.getenv('SLURM_JOB_ID'))
    if mode=='fresh':
        import cupy as cp
        from .conventional import cocheb
        from .tdvp import tdvp_response
        from .problems import device
        problem=load_reference(case)
        metadata.update(device(),cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    for method in methods:
        print(f'{case}: {method} ({mode})',flush=True)
        axis_reports=[]
        if mode=='replay':
            curves[method]=source[method].copy()
            seconds=oldtimes[method]
        else:
            values=[]
            for axis in spec['axes']:
                np.random.seed(20261103)
                cp.random.seed(20261103)
                broadening=spec['sigma_cm'] if case=='IR' else spec['eta_scaled']
                if method=='tangent1':
                    result=run_tangent(problem,axis,case=case,actions=spec['tache_actions'])
                elif method=='cocheb':
                    result=cocheb(problem,axis,case=case,vectors=spec['cocheb_vectors'],grid=grid,broadening=broadening)
                else:
                    result=tdvp_response(problem,axis,case=case,dt=spec['tdvp_dt'],
                        steps=spec['tdvp_steps'],grid=grid,broadening=broadening)
                np.testing.assert_allclose(result['grid'],grid,rtol=0,atol=1e-12)
                values.append(result['spectrum'])
                folder=output/method/axis
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'result.npz',**{
                    k:v for k,v in result.items() if isinstance(v,np.ndarray)
                    and (save_matrices or k not in {'overlap','projected_hamiltonian_scaled'})})
                diagnostic=_jsonable_result(result)
                diagnostic['reference_sha256']=problem.preparation['source_reference_sha256']
                (folder/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                axis_reports.append(diagnostic)
            curves[method]=sum(values)
            seconds=sum(sum(r['stage_seconds'].values()) for r in axis_reports)
        difference=float(np.linalg.norm(curves[method]-source[method])/np.linalg.norm(source[method]))
        error=float(100*np.linalg.norm(curves[method]-reference)/np.linalg.norm(reference))
        rows.append(dict(method=method,seconds=seconds,published_seconds=oldtimes[method],
            relative_difference_from_publication=difference,reference_relative_L2_percent=error,
            axes=axis_reports))
        print(f'  response {seconds:.3f} s; relative difference from saved curve {difference:.3e}',flush=True)
    from .plotting import plot_comparison
    times={r['method']:r['seconds'] for r in rows}
    figure=plot_comparison(case,grid,reference,curves,times,output/f'{case.lower()}_spectrum_runtime.pdf')
    replay_figure=plot_comparison(case,grid,reference,{m:source[m] for m in methods},oldtimes,
        output/f'{case.lower()}_published_replay.pdf')
    np.savetxt(output/'spectra.csv',np.column_stack([grid,reference]+[curves[m] for m in methods]),
        delimiter=',',header='frequency,reference,'+','.join(methods),comments='')
    fields=['method','seconds','published_seconds','relative_difference_from_publication','reference_relative_L2_percent']
    with (output/'timings.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    report=dict(case=case,mode=mode,settings=spec,rows=rows,environment=metadata,
        figure=str(figure),published_replay_figure=str(replay_figure),
        elapsed_wall_seconds=time.time()-started,reference_preparation_reused=True,
        timing_policy='Synchronized response generation plus final processing; excludes model/reference setup, scaling, warmup, I/O and plotting.',
        status='completed',maximum_relative_spectrum_difference=max(r['relative_difference_from_publication'] for r in rows))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def reproduce_ir(**kwargs):
    """Notebook block 1: CH3CN IR D16; forwards keyword arguments to reproduce."""
    return reproduce('IR',**kwargs)


def reproduce_emission(**kwargs):
    """Notebook block 2: L10, 298 K fast-mode thermal emission at D32."""
    return reproduce('fast',**kwargs)
