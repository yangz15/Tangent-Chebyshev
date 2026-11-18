"""Physical/auxiliary purification helpers."""
import numpy as np
import cupy as cp
from renormalizer import Op
from renormalizer.model import Phonon
from renormalizer.tn import TTNO
from renormalizer.utils import Quantity
from renormalizer.mps.svd_qn import add_outer
from renormalizer.tn.contractions import bounded_contract
from .gpu_charged import device_operator
def phonon(omega,s,nbas):
    w=Quantity(omega,'cm-1')
    return Phonon.simple_phonon(w,Quantity(np.sqrt(2*s/w.as_au())),nbas)


def apply_device(operator,state):
    new=state.copy()
    for old,node,onode in zip(state,new,operator):
        si=state.get_node_indices(old,ttno=operator);oi=operator.get_node_indices(onode)
        out=[];shape=[];nc=len(old.children)
        for k in range(nc):shape.append(old.shape[k]*onode.shape[k]);out.extend([si[k],oi[k]])
        b=state.tn2bn[old]
        for k in range(b.n_sets):shape.append(old.shape[nc+k]);out.append(('up',str(b.dofs[k])))
        shape.append(old.shape[-1]*onode.shape[-1]);out.extend([si[-1],oi[-1]])
        node.tensor=bounded_contract(old.tensor,si,onode.tensor,oi,out).reshape(shape)
        node.qn=add_outer(old.qn,onode.qn).reshape(shape[-1],state.basis.qn_size)
    return new


def seed_thermal(problem,state,tau=1e-6):
    """Populate hopping charge sectors before local PS2 thermal evolution.

    The beta=0 paired purification has only equal P/Q charges on every cut.
    Local two-node sweeps cannot generate the unequal sectors across the
    intervening phonon nodes. Apply (I-tau*Hphysical) globally once. The
    remaining imaginary-time interval must be beta/2-tau. Taylor error is
    O(tau**2), checked by varying tau and by an exact thermal oracle.
    """
    operator=device_operator(TTNO(problem['tree'],
        [Op('I',0,qn=[[0,0]])]+[-tau*t for t in problem['hp'].terms]))
    seeded=apply_device(operator,state)
    seeded.canonicalise();seeded.root.tensor/=cp.linalg.norm(seeded.root.tensor)
    return seeded
