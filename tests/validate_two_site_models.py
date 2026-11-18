"""Exercise full model two-site TaChe plus Burg, independently of target plots."""
import json
import os
from pathlib import Path
import numpy as np
from ttns_spectra import load_reference,run_tangent
from ttns_spectra.paths import DATA

out=Path('outputs/two_site_checks');out.mkdir(parents=True,exist_ok=True)
rows=[]
for case,axis,actions in [('IR','x',250),('fast','response',75)]:
    problem=load_reference(case)
    result=run_tangent(problem,axis,case=case,actions=actions,two_site=True)
    with np.load(DATA/f'published/{case}/tangent2/{axis}/moments.npz') as f:
        expected=f['moments'][:2*actions]
    relative=float(np.linalg.norm(result['moments']-expected)/np.linalg.norm(expected))
    assert relative<1e-8,relative
    assert np.isfinite(result['spectrum']).all()
    np.savez_compressed(out/f'{case.lower()}_{axis}.npz',grid=result['grid'],
        spectrum=result['spectrum'],moments=result['moments'])
    row=dict(case=case,axis=axis,actions=actions,computed_coefficients=len(result['moments']),
        two_site=result['two_site'],bounds_cached=result['bounds_cached'],
        moment_relative_difference=relative,reconstruction=result['reconstruction'],
        stage_seconds=result['stage_seconds'])
    rows.append(row)
    print(json.dumps(row),flush=True)
report=dict(status='passed',job=os.getenv('SLURM_JOB_ID'),rows=rows,
    scope='Supplementary IR x and fast two-site TaChe/Burg checks; target PDFs remain one-site comparisons')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
