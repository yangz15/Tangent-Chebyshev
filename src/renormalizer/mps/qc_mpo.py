"""Memory-bounded real electronic MPO construction in spatial orbitals.

Normal-ordered fermionic monomials are encoded as uint16 local-operator
tables in batches. No spin-orbital rank-four tensor or O(N**4) list of Op
objects is retained. Sparse graph factorization avoids dense coefficient QR;
numeric MPO tensors are assembled without a dense grid of Python lists.
"""
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
from renormalizer import Model, Mpo, Op
from renormalizer.model.basis import BasisSpatialOrbital
from renormalizer.mps.symbolic_mpo import OpTuple, _construct_symbolic_mpo


def spatial_model(norb):
    return Model([BasisSpatialOrbital((2*i,2*i+1)) for i in range(norb)],[])


def _local_algebra():
    matrices=np.array([np.eye(2),np.diag([1,-1]),[[0,1],[0,0]],
                       [[0,0],[1,0]],np.diag([0,1]),np.diag([1,0])],dtype=np.int8)
    codes=np.zeros((6,6),dtype=np.uint8)
    signs=np.zeros((6,6),dtype=np.int8)
    for i in range(6):
        for j in range(6):
            value=matrices[i]@matrices[j]
            for k,matrix in enumerate(matrices):
                if np.array_equal(value,matrix):
                    codes[i,j],signs[i,j]=k,1
                    break
                if np.array_equal(value,-matrix):
                    codes[i,j],signs[i,j]=k,-1
                    break
            if np.any(value) and signs[i,j]==0:
                raise AssertionError("local algebra is not closed")
    return codes,signs


def _encode(indices,symbols,norb):
    """Right-multiply JW strings in their original fermion-operator order."""
    count=len(indices)
    local=np.zeros((count,2*norb),dtype=np.uint8)
    factors=np.ones(count,dtype=np.int8)
    codes,signs=_local_algebra()
    sites=np.arange(2*norb)[None,:]
    rows=np.arange(count)
    for column,symbol in enumerate(symbols):
        orbital=indices[:,column]
        mask=sites<orbital[:,None]
        factors *= np.prod(np.where(mask,signs[local,1],1),axis=1,dtype=np.int8)
        local=np.where(mask,codes[local,1],local)
        factors *= signs[local[rows,orbital],symbol]
        local[rows,orbital]=codes[local[rows,orbital],symbol]
    spatial=local[:,::2].astype(np.uint16)*6+local[:,1::2]
    return spatial+36*np.arange(norb,dtype=np.uint16)[None,:],factors


def _primary_ops(norb):
    result=[]
    symbols=("I","Z","+","-","- +","+ -")
    for site in range(norb):
        for alpha in range(6):
            for beta in range(6):
                pieces=[]
                for spin,code in enumerate((alpha,beta)):
                    qns=[]
                    for token in symbols[code].split():
                        qn=[0,0]
                        qn[spin]=1 if token=="-" else -1 if token=="+" else 0
                        qns.append(qn)
                    pieces.append(Op(symbols[code],2*site+spin,qn=qns))
                result.append(Op.product(pieces))
    return result


def electronic_table(h1,eri):
    """Chemists' real integrals -> unique normal-ordered JW product rows."""
    h1,eri=np.asarray(h1),np.asarray(eri)
    n=len(h1)
    if np.iscomplexobj(h1) or np.iscomplexobj(eri) or eri.shape!=(n,n,n,n):
        raise ValueError("real square spatial integrals required")
    if 36*n>=65536:
        raise ValueError("too many local operator IDs for uint16")
    pairs=np.array(list(combinations(range(2*n),2)),dtype=int)
    # Only pairs in the same total alpha/beta sector can couple.
    sectors=[pairs[(pairs%2).sum(axis=1)==k] for k in (0,1,2)]
    capacity=2*n*n+sum(len(p)**2 for p in sectors)
    table=np.empty((capacity,n),dtype=np.uint16)
    factor=np.empty(capacity,dtype=float)
    cursor=0
    p,q=np.nonzero(h1)
    for spin in (0,1):
        indices=np.column_stack((2*p+spin,2*q+spin))
        rows,sign=_encode(indices,(3,2),n)
        length=len(rows)
        table[cursor:cursor+length]=rows
        factor[cursor:cursor+length]=h1[p,q]*sign
        cursor+=length
    for sector in sectors:
        r,s=sector.T
        for p,q in sector:
            value=((p%2==s%2)&(q%2==r%2))*eri[p//2,s//2,q//2,r//2]
            value=value-((p%2==r%2)&(q%2==s%2))*eri[p//2,r//2,q//2,s//2]
            keep=np.flatnonzero(value)
            for start in range(0,len(keep),2048):
                chosen=keep[start:start+2048]
                indices=np.column_stack((np.full(len(chosen),p),np.full(len(chosen),q),r[chosen],s[chosen]))
                rows,sign=_encode(indices,(3,3,2,2),n)
                good=sign!=0
                length=int(good.sum())
                table[cursor:cursor+length]=rows[good]
                factor[cursor:cursor+length]=value[chosen][good]*sign[good]
                cursor+=length
    return table[:cursor].copy(),factor[:cursor].copy(),_primary_ops(n)


def electronic_mpo(model,h1,eri,progress=print):
    started=time.perf_counter()
    table,factor,primary=electronic_table(h1,eri)
    if not len(table):
        return Mpo.identity(model).scale(0.)
    if progress:
        progress(f"QC compact table: {len(table)} terms, {table.nbytes/2**20:.1f} MiB; sparse graph decomposition",flush=True)
    boundary=np.zeros((len(table),1),dtype=np.uint16)
    table=np.concatenate((boundary,table,boundary),axis=1)
    initial=[[OpTuple([0],qn=np.zeros(2,dtype=int),factor=1.)]]
    outgoing=_construct_symbolic_mpo(table,initial,factor,primary,"Hopcroft-Karp")
    del table,factor
    mpo=Mpo()
    mpo.model=model
    mpo.offset=0.
    mpo.to_right=False
    mpo.qn=[np.array([state[0].qn for state in layer],dtype=int) for layer in outgoing]
    mpo.qntot=mpo.qn[-1][0].copy()
    mpo.qn[-1]=np.zeros((1,2),dtype=int)
    mpo.qnidx=len(model.basis)-1
    for site in range(len(model.basis)):
        numeric=np.zeros((len(outgoing[site]),4,4,len(outgoing[site+1])),dtype=float)
        matrices={}
        for right,entries in enumerate(outgoing[site+1]):
            for term in entries:
                left,op_index=term.symbol
                if op_index not in matrices:
                    matrices[op_index]=model.basis[site].op_mat(primary[op_index])
                numeric[left,:,:,right] += term.factor*matrices[op_index]
        mpo.append(numeric)
        if progress:
            progress(f"QC tensor {site+1}/{len(model.basis)}: {numeric.shape}, cumulative {mpo.total_bytes/2**30:.3f} GiB",flush=True)
    if progress:
        progress(f"QC MPO complete in {time.perf_counter()-started:.1f} s",flush=True)
    return mpo


def save_numeric_mpo(mpo,directory):
    """One file per site; load can memory-map without retaining symbolic data."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for i,mt in enumerate(mpo):
        temporary=directory/f"site_{i}.tmp.npy"
        np.save(temporary,mt.array)
        temporary.replace(directory/f"site_{i}.npy")
    metadata=dict(nsites=len(mpo),qn=[q.tolist() for q in mpo.qn],qnidx=mpo.qnidx,
                  qntot=mpo.qntot.tolist(),to_right=bool(mpo.to_right),offset=0.)
    temp=directory/"metadata.tmp"
    temp.write_text(json.dumps(metadata)+"\n")
    temp.replace(directory/"metadata.json")


def load_numeric_mpo(model,directory):
    directory=Path(directory)
    meta=json.loads((directory/"metadata.json").read_text())
    mpo=Mpo()
    mpo.model=model
    for i in range(meta["nsites"]):
        mpo.append(np.load(directory/f"site_{i}.npy",mmap_mode="r"))
    mpo.qn=[np.asarray(q,dtype=int) for q in meta["qn"]]
    mpo.qntot=np.asarray(meta["qntot"],dtype=int)
    mpo.qnidx,mpo.to_right,mpo.offset=meta["qnidx"],meta["to_right"],meta["offset"]
    return mpo
