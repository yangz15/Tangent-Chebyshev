"""Verify executed notebook, inspect outputs, and write reproducibility evidence."""
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import nbformat

root=Path(__file__).resolve().parents[1]
out=root/'outputs'
sys.path.insert(0,str(root/'src'))
from ttns_spectra.paths import DATA
checks={}
input_path=DATA/'molecular/ch3cn/vib.out'
assert hashlib.sha256(input_path.read_bytes()).hexdigest()=='047f19b153d23dab2c8a0cfdbc60d310da0530fd96a82a57a595ffd5a41a3c73'
checks['required_molecular_input_present_and_verified']=True
for path in (root/'src').rglob('*.py'):
    compile(path.read_text(),str(path),'exec')
    if '/curie-home/' in path.read_text():
        raise AssertionError(f'Nonportable runtime path: {path}')
checks['python_sources_compile']=True
nb=nbformat.read(root/'main.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert len(code)==3
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert all(not any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in ast.parse(c.source).body) for c in code)
checks['notebook_setup_and_two_blocks_executed']=True
reports=[json.loads((out/case/'report.json').read_text()) for case in ('ir','fast')]
for report in reports:
    assert report['status']=='completed' and report['mode']=='fresh'
    assert report['reference_preparation_reused']
    for row in report['rows']:
        assert row['seconds']>0
        # Refactoring should reproduce the saved spectrum, while exact timing
        # and floating-point contraction paths need not match bit for bit.
        assert row['relative_difference_from_publication']<1e-5,(report['case'],row)
        for axis in row['axes']:
            if row['method']=='cocheb':
                assert all(f['converged'] for f in axis['fits'])
                assert axis['computed_coefficients']==report['settings']['cocheb_vectors']
            if row['method']=='tangent1':
                assert axis['actions']==report['settings']['tache_actions']
                assert axis['computed_coefficients']==report['settings']['computed_coefficients']
                assert axis['bounds_cached'] and not axis['response_eigenproblem']
    arrays=np.genfromtxt(out/report['case'].lower()/'spectra.csv',delimiter=',',names=True)
    assert all(np.isfinite(arrays[n]).all() for n in arrays.dtype.names)
checks['all_six_fresh_spectra_match_saved_below_1e_minus_5']=True
core=json.loads((out/'core_validation.json').read_text())
dense=json.loads((out/'tangent_validation.json').read_text())
assert core['status']==dense['status']=='passed'
checks['neutral_charged_one_two_site_dense_oracles_passed']=True
bounds=json.loads((out/'bounds_validation.json').read_text())
two_site=json.loads((out/'two_site_checks/report.json').read_text())
portable=json.loads((out/'portability.json').read_text())
assert bounds['status']==two_site['status']==portable['status']=='passed'
checks['adaptive_endpoint_helper_matches_dense_extrema']=True
checks['two_site_TaChe_Burg_on_both_models_passed']=True
checks['relocated_wheel_and_CPU_replay_passed']=True
environment=dict(python=sys.version,platform=platform.platform(),packages={})
for name in ('numpy','scipy','h5py','opt_einsum','sympy','matplotlib','cupy-cuda12x','cupy-cuda11x','cupy','nbformat','nbclient','ipykernel','jupyterlab','setuptools'):
    try:environment['packages'][name]=metadata.version(name)
    except metadata.PackageNotFoundError:pass
(out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
(root/'constraints-tested.txt').write_text(
    '# Direct-package versions from the tested Python '+platform.python_version()+' environment.\n'
    '# This is not a complete transitive lock file. Choose the CuPy wheel for your CUDA version.\n'
    +'\n'.join(name+'=='+version for name,version in sorted(environment['packages'].items()))+'\n')
summary=dict(status='passed',checks=checks,
    fresh_notebook_jobs=sorted({r['environment']['slurm_job'] for r in reports}),
    core_validation=core,dense_validation_job=dense['job'],bounds_validation=bounds,
    two_site_model_validation=two_site,portability=portable,
    spectra=[dict(case=r['case'],method=x['method'],seconds=x['seconds'],
        published_seconds=x['published_seconds'],relative_difference=x['relative_difference_from_publication'],
        reference_relative_L2_percent=x['reference_relative_L2_percent']) for r in reports for x in r['rows']])
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
text='''# Validation performed for this delivery

The main notebook was executed in fresh mode on a V100 through Slurm. Both
requested calculation blocks completed. All three response methods were
regenerated from the bundled prepared references; their saved publication
spectra were used only for comparison. Independent high-accuracy reference
curves and endpoint scales were reused, as were the prepared ground/thermal
states. No fresh DMRG/thermal convergence claim is made.

| Case | Method | Fresh response s | Published s | Relative curve difference |
|---|---|---:|---:|---:|
'''
for x in summary['spectra']:
    text+=f"| {x['case']} | {x['method']} | {x['seconds']:.6f} | {x['published_seconds']:.6f} | {x['relative_difference']:.3e} |\n"
text+=f'''
Notebook Slurm job: {', '.join(summary['fresh_notebook_jobs'])}.
GPU core/dense checks: {core['job']} / {dense['job']}.

`tests/validate_core.py` checks Burg coefficients against a direct Toeplitz
solve, rejects invalid covariances, checks continuation input preservation,
compares the public one-site/two-site doubled-moment API to dense projected
recurrences, and compares cached coCheb S/H contractions to an independent
environment contraction implementation. The maximum doubled-moment error was
{max(core['moment_doubling_errors']):.3e}; actual-S/H error was {core['actual_SH_error']:.3e}.

`tests/dense_oracle.py` checks neutral and charged references on an eight-site
binary tree: overlap metric, Hermiticity, projected H action, source projection,
moments, and nesting of the one-site space in the enlarged two-site space.
All four cases passed. These are fresh checks, not copied old validation output.

`tests/validate_bounds.py` independently checks the new adaptive endpoint helper
against dense physical one-site/two-site extrema, including a small Krylov-rank
case. `tests/validate_two_site_models.py` additionally executes two-site TaChe
**with Burg reconstruction** for IR x (250 actions, 500 moments) and fast thermal
emission (75 actions, 150 moments). Computed moment prefixes agree with the
original two-site runs below 1e-8 relative error. Outputs are saved under
`outputs/two_site_checks`; these supplementary checks do not replace the
one-site comparisons in the requested figures.

A distribution wheel was built, installed into a separate temporary directory,
and used for both CPU figure replays without the original checkout on its import
path. That portability check passed; see `outputs/portability.json`.

`scripts/verify_release.py` checks source compilation/portability, notebook
schema and executed outputs, method/action counts, converged coCheb fits,
finite exported curves and fresh-vs-saved discrepancies below 1e-5.
Machine-readable evidence is in `outputs/verification.json`; software versions
are in `outputs/environment.json`.

Timings are single measurements and are not expected to reproduce the old
numbers exactly. Source fitting, model setup, reference preparation, endpoint
estimation, I/O, plotting and queue time are excluded. Gaussian reconstruction
timing includes an extra continuation consistency check. Equal coefficient
counts do not imply equal spectral accuracy. The requested two published panels
use one-site methods. The supplementary two-site model runs cover IR x and
aggregate response, rather than a complete new two-site convergence study.
'''
(root/'docs/VALIDATION.md').write_text(text)
# Portable installed-distribution metadata and all implementation definitions
# can be audited without reading a notebook's execution history.
inventory=[]
for path in sorted((root/'src/ttns_spectra').rglob('*.py')):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            inventory.append(dict(module=str(path.relative_to(root)),name=node.name,line=node.lineno,
                kind='class' if isinstance(node,ast.ClassDef) else 'function'))
(root/'docs/function_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
supplement='''

### Supplementary two-site runs actually performed

The release validation also ran `run_tangent(..., two_site=True)` with Burg
reconstruction on IR x and fast thermal emission. These were separate Slurm
validation runs; they are not represented as an extra executed notebook cell.

| Model | Projected actions | Computed moments | Relative moment difference |
|---|---:|---:|---:|
'''
for row in two_site['rows']:
    supplement+=f"| {row['case']} / {row['axis']} | {row['actions']} | {row['computed_coefficients']} | {row['moment_relative_difference']:.3e} |\n"
supplement+='\nFull arrays and diagnostics: `outputs/two_site_checks/`.\n'
marker='\n\n### Supplementary two-site runs actually performed'
nb.cells[-1].source=nb.cells[-1].source.split(marker)[0]+supplement
nbformat.write(nb,root/'main.ipynb')
print(json.dumps(summary,indent=2))
