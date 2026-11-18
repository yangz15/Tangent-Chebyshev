"""Numerical checks for covariance extension and the public TTNS response API."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz
from ttns_spectra import burg_maximum_entropy, extend_moments

out=Path('outputs');out.mkdir(exist_ok=True)
nodes=np.array([-.72,-.12,.23,.81]);weights=np.array([.1,.2,.4,.3])
mu=np.cos(np.arange(8)[:,None]*np.arccos(nodes))@weights
loading=1e-5
prediction,info=burg_maximum_entropy(mu,loading)
r=mu.copy();r[0]*=1+loading
oracle=solve_toeplitz(r[:-1],r[1:])
error=float(np.linalg.norm(prediction-oracle))
assert error<1e-9,error
assert info['prediction_variance']>0 and info['largest_reflection']<1
continued=extend_moments(mu,prediction,40)
assert np.array_equal(continued[:len(mu)],mu)
for invalid in ([0.,1.],[1.,2.]):
    try:burg_maximum_entropy(invalid,loading=0)
    except ValueError:pass
    else:raise AssertionError('Invalid covariance accepted')
report=dict(status='passed',toeplitz_prediction_error=error,job=os.getenv('SLURM_JOB_ID'))

if os.getenv('TTNS_GPU_TEST')=='1':
    import cupy as cp
    from renormalizer import BasisHalfSpin,Op
    from renormalizer.tn import TTNS,TTNO
    from ttns_spectra.topology import balanced_tree
    from ttns_spectra.tangent import TreeTangent
    from ttns_spectra.projected import CachedTreeBasis,flatten,flatten_operator
    from ttns_spectra._backend.gpu_charged import ChargedState,device_operator,matrix_element
    from ttns_spectra import tangent_chebyshev_1site,tangent_chebyshev_2site
    np.random.seed(15);cp.random.seed(15)
    tree=balanced_tree([BasisHalfSpin(i) for i in range(6)])
    ref=ChargedState.from_host(TTNS.random(tree,0,2),2).canonicalise().to_complex()
    ref.root.tensor/=cp.linalg.norm(ref.root.tensor)
    h=TTNO(tree,[Op('sigma_z',i,factor=.02*(i+1)) for i in range(6)])
    d=TTNO(tree,[Op('sigma_x',i,factor=.2) for i in range(6)])
    physical=[b for n in tree for b in n.basis_sets if b.nbas>1]
    hd=cp.asarray(h.todense(physical)).reshape(64,64)
    moment_errors=[]
    for two,api in [(False,tangent_chebyshev_1site),(True,tangent_chebyshev_2site)]:
        result=api(ref,h,d,bounds=[-1.,1.],actions=8,target=[0],remove_reference=True)
        space=TreeTangent(ref,[0],two_site=two,remove_reference=True)
        cols=[]
        for _ in range(96):
            v=space.zeros()
            for block in v.blocks:block[:]=cp.random.normal(size=block.shape)+1j*cp.random.normal(size=block.shape)
            cols.append(space.dense(v.restore_gauge(inplace=True)))
        u,s,_=cp.linalg.svd(cp.stack(cols,axis=1),full_matrices=False)
        q=u[:,s>1e-9*s[0]];p=q@q.conj().T
        source=space.dense(space.project(device_operator(d),ref))
        old=None;current=source.copy();expected=[]
        for n in range(16):
            expected.append(float(cp.vdot(source,current).real))
            nxt=p@hd@current if n==0 else 2*p@hd@current-old
            old,current=current,nxt
        difference=float(np.max(abs(result['moments']-expected)))
        assert difference<1e-9,difference
        moment_errors.append(difference)
    # Test cross contractions, including a partial cache batch, against the
    # independent environment matrix_element implementation.
    states=[ref]
    for _ in range(3):
        a=ChargedState.from_host(TTNS.random(tree,0,2),2).canonicalise().to_complex()
        states.append(a)
    cached=CachedTreeBasis(ref,batch=2,cache_bytes=1024**2)
    vectors=[[cp.asnumpy(t) for t in flatten(a)] for a in states]
    sc,hc=cached.column_gpu(vectors,flatten(states[-1]),flatten_operator(device_operator(h)))
    ident=device_operator(TTNO.identity(tree));hop=device_operator(h)
    so=cp.asarray([matrix_element(a,ident,states[-1]) for a in states])
    ho=cp.asarray([matrix_element(a,hop,states[-1]) for a in states])
    sh_error=max(float(cp.max(abs(sc-so))),float(cp.max(abs(hc-ho))))
    assert sh_error<1e-10,sh_error
    report.update(moment_doubling_errors=moment_errors,actual_SH_error=sh_error,
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
(out/'core_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
