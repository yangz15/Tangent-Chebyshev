"""Exact cached batched contractions of the actual compressed TTNS vectors."""
from functools import lru_cache
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import cupy as cp
import opt_einsum as oe
USE_GPU = True
def planned_expression(equation, shapes):
    return oe.contract_expression(equation, *shapes, optimize='optimal')


def contract(arrays, labels, output):
    equation=','.join(''.join(oe.get_symbol(i) for i in legs) for legs in labels)
    equation+='->'+''.join(oe.get_symbol(i) for i in output)
    return planned_expression(equation, tuple(a.shape for a in arrays))(
        *arrays,backend='cupy' if USE_GPU else 'numpy')


class ExactProjectedBasis:
    def __init__(self, engine, size=32, mode="implicit"):
        self.e, self.size = engine, size
    def elements(self,vectors,ket):
        result=cp.empty(len(vectors),dtype=cp.complex128)
        for ids,bra in self.blocks(vectors):
            env={}
            for i in self.e.postorder:
                kids=self.e.children[i];d=len(kids)
                b=list(range(d));k=list(range(d,2*d));physical,bp,kp,batch=range(2*d,2*d+4)
                arrays=[bra[i],ket[i]];labels=[[batch]+b+[physical,bp],k+[physical,kp]]
                for j,child in enumerate(kids):
                    arrays.append(env.pop(child));labels.append([batch,b[j],k[j]])
                env[i]=contract(arrays,labels,[batch,bp,kp])
            result[cp.asarray(ids)]=env[self.e.root][:,0,0]
        return result

    def operator_elements(self,vectors,ket,operator):
        result=cp.empty(len(vectors),dtype=cp.complex128)
        for ids,bra in self.blocks(vectors):
            env={}
            for i in self.e.postorder:
                kids=self.e.children[i];d=len(kids)
                b=list(range(d));k=list(range(d,2*d));o=list(range(2*d,3*d))
                pb,pk,bp,kp,op,batch=range(3*d,3*d+6)
                arrays=[bra[i],ket[i],operator[i]]
                labels=[[batch]+b+[pb,bp],k+[pk,kp],o+[pb,pk,op]]
                for j,child in enumerate(kids):
                    arrays.append(env.pop(child));labels.append([batch,b[j],o[j],k[j]])
                env[i]=contract(arrays,labels,[batch,bp,op,kp])
            result[cp.asarray(ids)]=env[self.e.root][:,0,0,0]
        return result


def flatten(state):
    values=[]
    for node in state:
        nc=len(node.children)
        values.append(node.tensor.reshape(*node.shape[:nc],-1,node.shape[-1]))
    return values


def flatten_operator(operator):
    result=[]
    for node in operator:
        nc=len(node.children);ns=operator.tn2bn[node].n_sets
        axes=list(range(nc))+list(range(nc,nc+2*ns,2))+list(range(nc+1,nc+2*ns,2))+[node.tensor.ndim-1]
        d=int(np.prod(node.shape[nc:-1:2]))
        result.append(cp.asarray(node.tensor).transpose(axes).reshape(*node.shape[:nc],d,d,node.shape[-1]))
    return result


class CachedTreeBasis(ExactProjectedBasis):
    def __init__(self,reference,batch=8,cache_bytes=4*1024**3):
        engine=SimpleNamespace(children=[[reference.node_idx[c] for c in n.children] for n in reference],
            root=0,postorder=[reference.node_idx[n] for n in reference.postorder_list()])
        super().__init__(engine,size=batch,mode='implicit')
        self.cache=OrderedDict();self.cache_bytes=cache_bytes;self.used_bytes=0
    def blocks(self,vectors):
        for start in range(0,len(vectors),self.size):
            stop=min(start+self.size,len(vectors))
            if start in self.cache:
                groups=self.cache[start];self.cache.move_to_end(start)
            else:
                shapes={}
                for j in range(start,stop):shapes.setdefault(tuple(t.shape for t in vectors[j]),[]).append(j)
                groups=[(ids,[cp.stack([cp.asarray(vectors[j][i]).conj() for j in ids])
                    for i in range(len(self.e.children))]) for ids in shapes.values()]
                size=sum(a.nbytes for _,arrays in groups for a in arrays)
                if stop-start==self.size and size<=self.cache_bytes:
                    while self.cache and self.used_bytes+size>self.cache_bytes:
                        _,old=self.cache.popitem(last=False)
                        self.used_bytes-=sum(a.nbytes for _,arrays in old for a in arrays)
                    self.cache[start]=groups;self.used_bytes+=size
            yield from groups
    def column_gpu(self,vectors,ket,operator):
        return self.elements(vectors,ket),self.operator_elements(vectors,ket,operator)
