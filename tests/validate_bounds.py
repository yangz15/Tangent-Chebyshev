"""Check numerical endpoint setup on physical one-site/two-site support."""
import json
import os
from pathlib import Path
import numpy as np
import cupy as cp
from renormalizer import BasisHalfSpin, Op
from renormalizer.tn import TTNS, TTNO
from ttns_spectra.topology import balanced_tree
from ttns_spectra.tangent import TreeTangent
from ttns_spectra.chebyshev import estimate_tangent_bounds
from ttns_spectra._backend.gpu_charged import ChargedState, device_operator

np.random.seed(119);cp.random.seed(119)
tree=balanced_tree([BasisHalfSpin(i) for i in range(6)])
ref=ChargedState.from_host(TTNS.random(tree,0,2),2).canonicalise().to_complex()
ref.root.tensor/=cp.linalg.norm(ref.root.tensor)
operator=TTNO(tree,[Op('sigma_z',i,factor=.03*(i+1)) for i in range(6)]
              +[Op('sigma_x sigma_x',[i,i+1],factor=.02) for i in range(5)])
physical=[b for n in tree for b in n.basis_sets if b.nbas>1]
hd=cp.asarray(operator.todense(physical)).reshape(64,64)
rows=[]
for two in (False,True):
    space=TreeTangent(ref,[0],two_site=two,remove_reference=True)
    columns=[]
    for _ in range(96):
        v=space.zeros()
        for a in v.blocks:a[:]=cp.random.normal(size=a.shape)+1j*cp.random.normal(size=a.shape)
        columns.append(space.dense(v.restore_gauge(inplace=True)))
    u,s,_=cp.linalg.svd(cp.stack(columns,axis=1),full_matrices=False)
    q=u[:,s>s[0]*1e-9]
    exact=cp.asnumpy(cp.linalg.eigvalsh(q.conj().T@hd@q))[[0,-1]]
    estimated=estimate_tangent_bounds(space,device_operator(operator))
    print(dict(two_site=two, exact=exact.tolist(), estimate=estimated),flush=True)
    assert estimated[0] <= exact[0]+1e-8 and estimated[1] >= exact[1]-1e-8
    assert max(abs(np.asarray(estimated)-exact))<1e-3
    rows.append(dict(two_site=two,exact=exact.tolist(),padded_estimate=estimated))
result=dict(status='passed',job=os.getenv('SLURM_JOB_ID'),rows=rows)
Path('outputs/bounds_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
