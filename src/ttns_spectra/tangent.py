"""One-site plus orthogonal two-site tangent sectors on arbitrary trees.

All physical coordinates use orthonormal exterior frames. Internal contractions
use a common root gauge, with explicit small edge-frame conversions. A two-site
sector is discarded at BOTH ends of its edge. Directed insertion environments
sum all sectors in O(number of tree nodes) local networks per H action; networks
have at most three nodes. No Krylov matrix or history is stored.
"""
import numpy as np
import cupy as cp
from renormalizer.mps.svd_qn import add_outer
from renormalizer.tn.contractions import bounded_contract as contract


class Vector:
    def __init__(self, space, blocks):
        self.space, self.blocks = space, blocks
    def copy(self):
        return Vector(self.space, [x.copy() for x in self.blocks])
    def scale(self, value, inplace=False):
        result = self if inplace else self.copy()
        for x in result.blocks:
            x *= value
        return result
    def axpy(self, value, other):
        for x, y in zip(self.blocks, other.blocks):
            x += value*y
        return self
    def dot(self, other):
        return complex(sum(cp.vdot(x,y) for x,y in zip(self.blocks,other.blocks)))
    @property
    def norm(self):
        return np.sqrt(max(0.,self.dot(self).real))
    def restore_gauge(self, inplace=False):
        result = self if inplace else self.copy()
        result.blocks = [self.space.project_block(i,x) for i,x in enumerate(result.blocks)]
        return result


class TreeTangent:
    def __init__(self, reference, target=None, two_site=False, remove_reference=False,
                 frame_rcond=1e-13):
        self.reference = reference.copy().canonicalise()
        self.reference.root.tensor /= cp.linalg.norm(self.reference.root.tensor)
        ref = self.reference
        self.n = len(ref.node_list)
        self.parents = [None if n.parent is None else ref.node_idx[n.parent] for n in ref]
        self.children = [[ref.node_idx[c] for c in n.children] for n in ref]
        self.neighbors = [c+([] if p is None else [p]) for p,c in zip(self.parents,self.children)]
        self.a = [n.tensor.copy() for n in ref]
        self.target = np.asarray(ref.qntot if target is None else target)
        self.remove_reference = remove_reference
        self.two_site = two_site
        self.clusters = [(i,) for i in range(self.n)]
        if two_site:
            self.clusters += [(i,self.parents[i]) for i in range(1,self.n)]
        self.key = {frozenset(s):i for i,s in enumerate(self.clusters)}
        self.frames, self.support, self.centers, self.right, self.frame_diagnostics = [],[],[],{},[]
        self.center_qn = []
        for i in range(self.n):
            state = ref.copy()
            path=[];j=i
            while self.parents[j] is not None:
                path.append(j);j=self.parents[j]
            for child in reversed(path):
                parent=self.parents[child]
                state.push_cano_to_child(state.node_list[parent],self.children[parent].index(child))
                if child==i:
                    self.right[i]=state.node_list[parent].tensor.copy()
            center=state.node_list[i]
            self.centers.append(center.tensor.copy())
            self.center_qn.append(center.qn.copy())
            if i==0:
                self.frames.append(cp.ones((1,1)))
                self.support.append(cp.ones((1,1)))
                continue
            left=self.a[i].reshape(-1,self.a[i].shape[-1])
            transfer=left.conj().T@center.tensor.reshape(left.shape[0],-1)
            u,s,vh=cp.linalg.svd(transfer,full_matrices=False)
            keep=s>frame_rcond*s[0]
            inverse=(vh[keep].conj().T/s[keep])@u[:,keep].conj().T
            self.frames.append(inverse)
            self.support.append(vh[keep].conj().T@vh[keep])
            self.frame_diagnostics.append(dict(node=i,rank=int(keep.sum()),dimension=len(s),
                smallest_relative_singular=float(s[-1]/s[0]),
                conversion_residual=float(cp.linalg.norm(inverse@transfer-self.support[-1]))))
        self.shapes=[];self.masks=[];self.labels=[]
        for cluster in self.clusters:
            if len(cluster)==1:
                i=cluster[0]
                shape=self.centers[i].shape
                labels=self.node_labels(i,'b')
                q=np.zeros(ref.basis.qn_size,dtype=int)
                for child in self.children[i]:q=add_outer(q,ref.node_list[child].qn)
                for b in ref.tn2bn[ref.node_list[i]].basis_sets:q=add_outer(q,b.sigmaqn)
                q=add_outer(q,ref.qntot-self.center_qn[i])
            else:
                child,parent=cluster;axis=self.children[parent].index(child)
                shape=self.a[child].shape[:-1]+self.centers[parent].shape[:axis]+self.centers[parent].shape[axis+1:]
                labels=self.node_labels(child,'b')[:-1]+[x for j,x in enumerate(self.node_labels(parent,'b')) if j!=axis]
                q=np.zeros(ref.basis.qn_size,dtype=int)
                for kid in self.children[child]:q=add_outer(q,ref.node_list[kid].qn)
                for b in ref.tn2bn[ref.node_list[child]].basis_sets:q=add_outer(q,b.sigmaqn)
                for kid in self.children[parent]:
                    if kid!=child:q=add_outer(q,ref.node_list[kid].qn)
                for b in ref.tn2bn[ref.node_list[parent]].basis_sets:q=add_outer(q,b.sigmaqn)
                q=add_outer(q,ref.qntot-self.center_qn[parent])
            self.shapes.append(shape);self.labels.append(labels)
            self.masks.append(cp.asarray(np.all(q==self.target,axis=-1)))
            assert shape==self.masks[-1].shape,(cluster,shape,self.masks[-1].shape)
        self.dimension=sum(int(np.prod(s)) for s in self.shapes)
        self.cache={}

    def node_labels(self,i,side):
        # All physical sets stay explicit; root parent has dimension one.
        sets=self.reference.tn2bn[self.reference.node_list[i]].basis_sets
        return [(side,'e',j) for j in self.children[i]]+[(side,'p',i,k) for k in range(len(sets))]+[(side,'e',i)]

    def op_labels(self,i):
        b,k=self.node_labels(i,'b'),self.node_labels(i,'k');nc=len(self.children[i])
        return [('w','e',j) for j in self.children[i]]+[y for pair in zip(b[nc:-1],k[nc:-1]) for y in pair]+[('w','e',i)]

    def edge(self,i,j):
        return i if self.parents[i]==j else j

    def zeros(self):
        return Vector(self,[cp.zeros(s,dtype=cp.complex128) for s in self.shapes])

    def project_block(self,index,value):
        cluster=self.clusters[index];top=cluster[-1]
        value=value*self.masks[index]
        value=(value.reshape(-1,value.shape[-1])@self.support[top]).reshape(value.shape)
        if len(cluster)==1:
            node=cluster[0]
            if node:
                left=self.a[node].reshape(-1,self.a[node].shape[-1])
                mat=value.reshape(left.shape[0],-1)
                value=(mat-left@(left.conj().T@mat)).reshape(value.shape)
            elif self.remove_reference:
                value-=self.a[0]*cp.vdot(self.a[0],value)
        else:
            child,parent=cluster
            left=self.a[child].reshape(-1,self.a[child].shape[-1])
            axis=self.children[parent].index(child)
            right=cp.moveaxis(self.right[child],axis,0).reshape(self.right[child].shape[axis],-1)
            mat=value.reshape(left.shape[0],right.shape[1])
            mat-=left@(left.conj().T@mat)
            mat-=(mat@right.conj().T)@right
            value=mat.reshape(value.shape)
        return value*self.masks[index]

    def common(self,vector):
        return [(b.reshape(-1,b.shape[-1])@self.frames[s[-1]]).reshape(*b.shape[:-1],self.frames[s[-1]].shape[-1])
                for b,s in zip(vector.blocks,self.clusters)]

    def adjoint(self,index,value):
        top=self.clusters[index][-1]
        v=value.reshape(-1,value.shape[-1])@self.frames[top].conj().T
        return self.project_block(index,v.reshape(self.shapes[index]))

    def network(self,nodes,operator,zero,ket=None,insert=None,blocks=None,open_index=None,cut=None,override=None):
        nodes=set(nodes);args=[]
        if 0 in nodes:
            args.extend((cp.ones((1,1,1)),[('b','e',0),('w','e',0),('k','e',0)]))
        open_nodes=set() if open_index is None else set(self.clusters[open_index])
        inserted=set() if insert is None else set(self.clusters[insert])
        for i in sorted(nodes):
            args.extend((operator[i],self.op_labels(i)))
            if i not in open_nodes:args.extend((self.a[i].conj(),self.node_labels(i,'b')))
            if i not in inserted:args.extend(((self.a if ket is None else ket)[i],self.node_labels(i,'k')))
            for j in self.neighbors[i]:
                if j in nodes or (cut is not None and (i,j)==cut):continue
                edge=self.edge(i,j)
                value=override[1] if override is not None and override[0]==(j,i) else zero(j,i)
                args.extend((value,[('b','e',edge),('w','e',edge),('k','e',edge)]))
        if insert is not None:
            args.extend((blocks[insert],[('k',)+x[1:] for x in self.labels[insert]]))
        if open_index is not None:
            output=self.labels[open_index]
        else:
            edge=self.edge(*cut)
            output=[('b','e',edge),('w','e',edge),('k','e',edge)]
        return contract(*args,output)

    def environments(self,operator,ket=None):
        tensors=[cp.asarray(n.tensor) for n in operator]
        cache={}
        def zero(i,j):
            if (i,j) not in cache:
                cache[i,j]=self.network([i],tensors,zero,ket=ket,cut=(i,j))
            return cache[i,j]
        for i in range(1,self.n):
            zero(i,self.parents[i]);zero(self.parents[i],i)
        return tensors,zero

    def project(self,operator,ket):
        tensors,zero=self.environments(operator,[cp.asarray(n.tensor) for n in ket])
        values=[]
        for i,s in enumerate(self.clusters):
            value=self.network(s,tensors,zero,ket=[n.tensor for n in ket],open_index=i)
            values.append(self.adjoint(i,value))
        return Vector(self,values)

    def apply(self,operator,vector,energy_shift=0.):
        key=id(operator)
        if key not in self.cache:self.cache[key]=(operator,*self.environments(operator))
        _,tensors,zero=self.cache[key]
        blocks=self.common(vector);cache={}
        def one(i,j):
            if (i,j) in cache:return cache[i,j]
            value=self.network([i],tensors,zero,insert=i,blocks=blocks,cut=(i,j))
            for k in self.neighbors[i]:
                if k==j:continue
                value+=self.network([i],tensors,zero,cut=(i,j),override=((k,i),one(k,i)))
                pair=self.key.get(frozenset((i,k)))
                if pair is not None:
                    value+=self.network([i,k],tensors,zero,insert=pair,blocks=blocks,cut=(i,j))
            cache[i,j]=value
            return value
        result=[]
        for index,cluster in enumerate(self.clusters):
            members=set(cluster);value=None
            for i in cluster:
                for j in self.neighbors[i]:
                    if j in members:continue
                    term=self.network(cluster,tensors,zero,open_index=index,override=((j,i),one(j,i)))
                    value=term if value is None else value+term
            inserts=set(cluster)
            if self.two_site:
                for i in cluster:
                    for j in self.neighbors[i]:inserts.add(self.key[frozenset((i,j))])
            for insertion in sorted(inserts):
                nodes=members|set(self.clusters[insertion])
                term=self.network(nodes,tensors,zero,insert=insertion,blocks=blocks,open_index=index)
                value=term if value is None else value+term
            value=self.adjoint(index,value)
            if energy_shift:value-=energy_shift*vector.blocks[index]
            result.append(value)
        return Vector(self,result)

    def dense(self,vector):
        """Small-system validation only: explicit embedding, never production."""
        blocks=self.common(vector);result=None
        output=[label for i in range(self.n) for label in self.node_labels(i,'k') if label[1]=='p']
        for index,cluster in enumerate(self.clusters):
            args=[]
            for i in range(self.n):
                if i not in cluster:args.extend((self.a[i],self.node_labels(i,'k')))
            args.extend((blocks[index],[('k',)+x[1:] for x in self.labels[index]]))
            term=contract(*args,output).ravel()
            result=term if result is None else result+term
        return result
