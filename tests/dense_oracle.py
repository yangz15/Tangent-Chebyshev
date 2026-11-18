"""Dense oracle for arbitrary binary-tree one-/two-site tangent projection."""
import json,os,time
from pathlib import Path
import numpy as np
import cupy as cp
from renormalizer import BasisHalfSpin,BasisSimpleElectron,Op
from renormalizer.tn import TTNS,TTNO
from ttns_spectra._backend.gpu_charged import ChargedState,device_operator
from ttns_spectra.topology import balanced_tree
from ttns_spectra.tangent import TreeTangent

HERE=Path(__file__).resolve().parents[1]/'outputs'
HERE.mkdir(exist_ok=True)
np.random.seed(381);cp.random.seed(381)
rows=[]
for charged in (False,True):
    sites=8;dimension=2**sites
    basis=[BasisSimpleElectron(i) if charged else BasisHalfSpin(i) for i in range(sites)]
    tree=balanced_tree(basis)
    if charged:
        terms=[Op(r'a^\dagger a',[i,i],factor=.013*(i+1)) for i in range(sites)]
        terms += [Op(r'a^\dagger a',[i,j],factor=.003) for i in range(sites) for j in range(sites) if abs(i-j)==1]
        dip=TTNO(tree,[Op('a',i,factor=1+.1*i) for i in range(sites)])
        refcharge,target=4,[3]
    else:
        terms=[Op('sigma_z',i,factor=.013*(i+1)) for i in range(sites)]
        terms += [Op('sigma_x sigma_x',[i,i+1],factor=.003) for i in range(sites-1)]
        dip=TTNO(tree,[Op('sigma_x',i,factor=1+.1*i) for i in range(sites)])
        refcharge,target=0,[0]
    # The embedding exposes physical legs in traversal order. Rerooting can
    # change that order; the dense oracle must use the identical permutation.
    physical_order=[b for node in tree for b in node.basis_sets if b.nbas>1]
    hop=TTNO(tree,terms);h=cp.asarray(hop.todense(physical_order)).reshape(dimension,dimension)
    d=cp.asarray(dip.todense(physical_order)).reshape(dimension,dimension)
    reference=ChargedState.from_host(TTNS.random(tree,refcharge,2),2).canonicalise().to_complex()
    reference.root.tensor/=cp.linalg.norm(reference.root.tensor)
    ground=cp.asarray(reference.host().todense(physical_order)).ravel()
    hdevice,di=device_operator(hop),device_operator(dip)
    projectors=[]
    for two in (False,True):
        started=time.perf_counter()
        space=TreeTangent(reference,target,two_site=two,remove_reference=not charged)
        def random():
            v=space.zeros()
            for block in v.blocks:block[:]=cp.random.normal(size=block.shape)+1j*cp.random.normal(size=block.shape)
            return v.restore_gauge(inplace=True)
        columns=cp.stack([space.dense(random()) for _ in range(96)],axis=1)
        u,s,_=cp.linalg.svd(columns,full_matrices=False)
        keep=s>1e-9*s[0];q=u[:,keep];proj=q@q.conj().T;projectors.append(proj)
        v,w=random(),random();hv,hw=space.apply(hdevice,v),space.apply(hdevice,w)
        dv,dw=space.dense(v),space.dense(w)
        norm_error=abs(v.dot(w)-complex(cp.vdot(dv,dw)))/max(v.norm*w.norm,1e-30)
        action_error=float(cp.linalg.norm(space.dense(hv)-proj@h@dv)/cp.linalg.norm(dv))
        hermiticity=abs(v.dot(hw)-hv.dot(w))/max(v.norm*w.norm,1e-30)
        source=space.project(di,reference)
        source_error=float(cp.linalg.norm(space.dense(source)-proj@d@ground))
        oracle=proj@d@ground;previous=None;current=source
        old=None;now=oracle;moment_error=0.
        for n in range(20):
            moment_error=max(moment_error,abs(source.dot(current)-complex(cp.vdot(oracle,now))))
            image=space.apply(hdevice,current)
            nxt=image if n==0 else image.scale(2).axpy(-1,previous)
            nex=proj@h@now if n==0 else 2*proj@h@now-old
            previous,current=current,nxt;old,now=now,nex
        row=dict(charged=charged,two_site=two,physical_rank=int(keep.sum()),
            norm_error=norm_error,action_error=action_error,hermiticity=hermiticity,
            source_error=source_error,moment_error=moment_error,seconds=time.perf_counter()-started,
            frame_diagnostics=space.frame_diagnostics)
        rows.append(row);print(json.dumps(row),flush=True)
        assert max(norm_error,action_error,hermiticity,source_error,moment_error)<1e-8,row
    nesting=float(cp.linalg.norm(projectors[1]@projectors[0]-projectors[0]))
    assert nesting<1e-9,nesting
    if not charged:assert rows[-1]['physical_rank']>rows[-2]['physical_rank'],rows[-2:]
report=dict(status='passed',job=os.getenv('SLURM_JOB_ID'),
    gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),rows=rows,
    explicit_dense_matrices_used_only_for_small_validation=True)
(HERE/'tangent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('BINARY TREE TANGENT VALIDATED',flush=True)
