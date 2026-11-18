from .paths import DATA as HERE
import contextlib,hashlib,json,os,time,logging
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import cupy as cp
from renormalizer import Op
from renormalizer.model import HolsteinModel,Mol
from renormalizer.tn import TTNS,TTNO,TreeNodeTensor,copy_connection
from renormalizer.utils import Quantity,constant
from ._backend.gpu_charged import (
    ChargedState,device_operator,matrix_element,ground_sweep,tdvp_step,variational_sum)
from ._backend.gpu_tree import DeviceEnvironment,local_expression
from ._backend.thermal import phonon,apply_device,seed_thermal
from ._backend.tdvp_one_site import tdvp_one_site
from .topology import balanced_tree,ir_tree,aggregate_tree,describe

UNIT=1400*constant.cm2au
STUDY=json.loads((HERE/'study.json').read_text())

logging.getLogger('renormalizer').setLevel(logging.WARNING)

def stamp():
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter_ns()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n')
    temporary.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def device():
    name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    return dict(gpu=name,job=os.getenv('SLURM_JOB_ID'),node=os.getenv('SLURMD_NODENAME'),dtype='complex128')
def save(state,path):state.host().dump(str(path))
def load(tree,path,bond):return ChargedState.from_host(TTNS.load(tree,str(path)),bond).to_complex()

class Timer:
    def __init__(self):self.ns={}
    @contextlib.contextmanager
    def stage(self,name):
        start=stamp()
        try:yield
        finally:self.ns[name]=self.ns.get(name,0)+stamp()-start

def build(case,bond):
    c=STUDY[case]
    if case=='IR':
        from .molecular import ch3cn
        p=ch3cn()
        tree=ir_tree(p.model.basis)
        p.tree=tree;p.h=TTNO(tree,p.model.ham_terms)
        p.dipoles={a:TTNO(tree,terms) for a,terms in p.dipole_terms.items()}
        p.identity=TTNO.identity(tree);p.old_energy=p.energy
        p.ground=None;p.bond=bond
        p.metadata=dict(p.metadata,topology=tree.topology_label,
            hierarchy_zero_based=tree.hierarchy,original_ground_used_for_response=False)
        return p
    ph=[phonon(1400,1,c['fast_basis'])]
    if case=='fast_slow':ph.append(phonon(140,5,c['slow_basis']))
    n=c['molecules'];j=np.zeros((n,n))
    for distance,value in ((1,700),(2,87.5)):
        for i in range(n-distance):j[i,i+distance]=j[i+distance,i]=value*constant.cm2au
    model=HolsteinModel([Mol(Quantity(0),ph,dipole=1.)]*n,j,scheme=2)
    leaves=[]
    for basis in model.basis:
        p=basis.copy(basis.dofs[0]);q=basis.copy(('Q',basis.dofs));charge=basis.sigmaqn[:,0]
        p.sigmaqn=np.column_stack((charge,np.zeros_like(charge)))
        q.sigmaqn=np.column_stack((np.zeros_like(charge),charge));leaves.append([p,q])
    for i in range(n):
        offset=i*(len(ph)+1)
        assert model.basis[offset].dofs==(i,)
        assert all(model.basis[offset+k+1].dofs==((i,k),) for k in range(len(ph)))
    tree=aggregate_tree(leaves,n,len(ph),c.get('local_tree_order','electron_fast'))
    def lift(term,aux=False):
        return Op(term.symbol,[('Q',(d,)) for d in term.dofs] if aux else term.dofs,
            factor=term.factor,qn=[[0,int(q[0])] if aux else [int(q[0]),0] for q in term.qn_list])
    pt=[lift(t) for t in model.ham_terms];qt=[lift(t,True) for t in model.ham_terms]
    hp=TTNO(tree,pt);h=TTNO(tree,pt+[-t for t in qt])
    dipole=TTNO(tree,[Op('a',i,qn=[[-1,0]]) for i in range(n)])
    identity=TTNO(tree,[Op('I',0,qn=[[0,0]])])
    nodes=[]
    for node in tree:
        if len(node.basis_sets)==1 and node.basis_sets[0].nbas==1:
            tensor=np.ones((1,)*len(node.children)+(1,1))
        else:
            b=node.basis_sets[0];d=b.nbas
            tensor=(np.eye(d)/np.sqrt(d) if b.is_phonon else np.diag([1.,0.])).reshape((1,)*len(node.children)+(d,d,1))
        nodes.append(TreeNodeTensor(tensor,np.zeros((1,2),int)))
    ref=TTNS(tree,root=copy_connection(tree.node_list,nodes))
    pair=TTNO(tree,[Op(r'a^\dagger a^\dagger',[i,('Q',(i,))],qn=[[1,0],[0,1]]) for i in range(n)])
    ref=apply_device(device_operator(pair),ChargedState.from_host(ref,bond)).canonicalise().to_complex()
    ref.root.tensor/=cp.linalg.norm(ref.root.tensor)
    return SimpleNamespace(model=model,tree=tree,h=device_operator(h),hp=device_operator(hp),
        dipoles={'response':device_operator(dipole)},identity=device_operator(identity),ground=ref,
        bond=bond,metadata=dict(molecules=n,j1_cm=700,j2_cm=87.5,temperature_K=298,
            fast_basis=c['fast_basis'],slow_basis=c['slow_basis'] if case=='fast_slow' else None,
            topology=tree.topology_label,hierarchy_zero_based=tree.hierarchy))

def fit(guess,terms,constraint=None,identity=None,tolerance=1e-10,max_sweeps=80):
    """Whole-target variational fit; optional physical ground constraint."""
    result=guess.copy().canonicalise();history=[]
    for sweep in range(max_sweeps):
        envs=[DeviceEnvironment(ket,op,bra=result) for _,op,ket in terms]
        ce=None if constraint is None else DeviceEnvironment(constraint,identity,bra=result)
        allenv=envs+([] if ce is None else [ce])
        def solve(node):
            index=result.node_idx[node]
            value=sum(coefficient*local_expression(env,index)(ket.node_list[index].tensor)
                for (coefficient,_,ket),env in zip(terms,envs))
            if ce is not None:
                ground=local_expression(ce,index)(constraint.node_list[index].tensor)
                norm=cp.vdot(ground,ground).real
                if float(norm)>1e-28:value-=ground*(cp.vdot(ground,value)/norm)
            node.tensor=value
        def visit(node):
            solve(node)
            for axis,child in enumerate(node.children):
                result.push_cano_to_child(node,axis)
                for env in allenv:env.down(result.node_idx[node],axis)
                visit(child);result.push_cano_to_parent(child)
                for env in allenv:env.up(result.node_idx[child])
                solve(node)
        visit(result.root)
        norm=float(cp.vdot(result.root.tensor,result.root.tensor).real)
        cross=sum(coefficient*complex(env.scalar()) for (coefficient,_,_),env in zip(terms,envs))
        objective=norm-2*cross.real;history.append(objective)
        change=np.inf if len(history)<2 else abs(history[-1]-history[-2])/max(norm,abs(objective),1e-300)
        if sweep>=1 and change<tolerance:
            overlap=0. if ce is None else abs(complex(ce.scalar()))/np.sqrt(max(norm,1e-300))
            assert overlap<1e-9,overlap
            return result,dict(converged=True,sweeps=sweep+1,objective=history,relative_change=change,
                ground_overlap=overlap,full_H_image_truncated_before_fit=False)
    raise RuntimeError(f'Whole-target variational fit unconverged: {sweep+1} sweeps, change={change}')

def source(p,axis,case):
    op=device_operator(p.dipoles[axis]);identity=device_operator(p.identity)
    mean=complex(matrix_element(p.ground,op,p.ground)) if case=='IR' else 0.
    terms=[(1.,op,p.ground)]
    if mean:terms.append((-mean,identity,p.ground))
    exact=apply_device(op,p.ground)
    guess=variational_sum(exact,terms,sweeps=2)
    guess.chi=p.bond
    fitted,info=fit(guess,terms,constraint=p.ground if case=='IR' else None,identity=identity)
    info.update(mean=[mean.real,mean.imag],weight=float(matrix_element(fitted,identity,fitted).real))
    return fitted,info

def prepare(case,bond,out):
    start=stamp();timer=Timer();c=STUDY[case];out.mkdir(parents=True,exist_ok=True)
    with timer.stage('model_setup'):p=build(case,bond)
    history=[]
    if case=='IR':
        with timer.stage('binary_ground_DMRG'):
            condition={b.dof:cp.asnumpy(cp.linalg.eigh(cp.asarray(b.op_mat('t')+.5*b.op_mat('q^2')))[1][:,0]) for b in p.model.basis}
            product=TTNS(p.tree,condition=condition)
            seed=product.add(TTNS.random(p.tree,0,4).scale(1e-3))
            state=ChargedState.from_host(seed,bond).to_complex().canonicalise()
            state.root.tensor/=cp.linalg.norm(state.root.tensor)
            op=device_operator(p.h);stable=0
            for sweep in range(40):
                try:
                    residual=ground_sweep(state,op,tolerance=1e-9,solver='davidson')
                    solver='davidson'
                except RuntimeError as exc:
                    if 'GPU Davidson' not in str(exc):raise
                    residual=ground_sweep(state,op,tolerance=1e-9,solver='lanczos')
                    solver='lanczos_fallback_same_tolerance'
                energy=float(matrix_element(state,op,state).real)
                change=abs(energy-history[-1]['energy']) if history else np.inf
                stable=stable+1 if change<1e-10 else 0
                history.append(dict(sweep=sweep+1,energy=energy,change=change,local_residual=residual,solver=solver))
                save(state,out/'reference.npz');write(out/'progress.json',dict(history=history))
                if stable>=3:break
            if stable<3:raise RuntimeError('Binary DMRG ground did not converge')
        p.ground=state;p.energy=energy
        bounds=[energy+v*constant.cm2au for v in c['cocheb_excitation_bounds_cm']]
        bounds_note='Previously selected common IR excitation interval; ground constraint retained. Not a newly certified full-H enclosure.'
    else:
        with timer.stage('binary_purified_thermal_preparation'):
            tau=c['thermal_seed_tau_au']
            state=seed_thermal(dict(hp=p.hp,tree=p.tree),p.ground,tau=tau)
            dt=-1j*(Quantity(c['temperature_K'],'K').to_beta()/2-tau)/c['thermal_steps']
            for step in range(c['thermal_steps']):
                info=tdvp_step(state,p.hp,dt,krylov_tol=1e-10,krylov_bytes=128*1024**2)
                state.root.tensor/=cp.linalg.norm(state.root.tensor)
                energy=float(matrix_element(state,p.hp,state).real)
                history.append(dict(step=step+1,energy=energy,**info))
                if (step+1)%5==0:
                    save(state,out/'thermal_checkpoint.npz');write(out/'progress.json',dict(history=history))
            p.ground=state
        oldcase='L10_fast_J2' if case=='fast' else 'L10_fast_slow_J2'
        oldprep=HERE/'full_bounds'/f'{case}.json'
        bounds=json.loads(oldprep.read_text())['bounds']
        bounds_note=f'Cached full physical response-H bounds, topology-independent Hamiltonian. Source: {oldprep}; sha256={sha(oldprep)}'
        p.energy=0.
    central_rank=p.ground.node_list[p.tree.central_node_index].shape[-1]
    assert central_rank==bond,('Central rank did not reach requested cap',central_rank,bond)
    save(p.ground,out/'reference.npz')
    summary=dict(status='completed',case=case,bond=bond,device=device(),metadata=p.metadata,
        energy=p.energy,bounds=bounds,bounds_note=bounds_note,history=history,
        topology=describe(p.tree,p.ground),stage_ns=timer.ns,full_ns=stamp()-start,
        physical_model_unchanged=True,real_time_response_TDVP=False,
        central_rank=central_rank,central_node_index=p.tree.central_node_index,
        topology_revision=p.tree.topology_label+'; redundant root suppressed; physical-leaf orientation',
        source_reference_sha256=sha(out/'reference.npz'))
    write(out/'summary.json',summary)
    return summary

def prepared(case,bond):
    folder=HERE/f'results/{case}/D{bond}/prepare'
    record=json.loads((folder/'summary.json').read_text());assert record['status']=='completed'
    assert sha(folder/'reference.npz')==record['source_reference_sha256']
    p=build(case,bond);p.ground=load(p.tree,folder/'reference.npz',bond);p.energy=record['energy']
    p.metadata=record['metadata'];p.preparation=record
    return p
