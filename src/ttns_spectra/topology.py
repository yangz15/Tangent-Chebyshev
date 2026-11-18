import numpy as np
from renormalizer.model.basis import BasisDummy
from renormalizer.tn import BasisTree,TreeNodeBasis

IR_HIERARCHY = (((0,(4,5)),((6,7),(8,9))),((2,(1,3)),(10,11)))

def balanced_hierarchy(items):
    items=list(items)
    if not items:raise ValueError('Empty hierarchy')
    if len(items)==1:return items[0]
    mid=len(items)//2
    return (balanced_hierarchy(items[:mid]),balanced_hierarchy(items[mid:]))

def hierarchy_tree(leaves,hierarchy,qn_size=1,label='explicit_binary'):
    """Explicit leaf grouping, oriented from a physical leaf.

    Suppress the redundant degree-two virtual root. Keeping it freezes the
    central Schmidt rank in two-node sweeps: each root bond is limited by the
    other root bond. All remaining virtual vertices have degree three, and
    every physical vertex has degree one. The physical root has one child;
    every virtual vertex has two children.
    """
    leaves=list(leaves);count=[0];used=[]
    def build(spec):
        if isinstance(spec,(int,np.integer)):
            if not 0<=spec<len(leaves):raise ValueError(('Invalid leaf',spec))
            used.append(int(spec))
            return TreeNodeBasis(leaves[spec])
        if len(spec)!=2:raise ValueError(('Expected binary branch',spec))
        count[0]+=1
        dummy=BasisDummy(('binary_virtual',count[0]),sigmaqn=np.zeros((1,qn_size),int))
        node=TreeNodeBasis(dummy)
        node.add_child([build(spec[0]),build(spec[1])])
        return node
    if len(leaves)<3:raise ValueError('At least three physical leaves required')
    oldroot=build(hierarchy)
    if sorted(used)!=list(range(len(leaves))):
        raise ValueError(('Every leaf must occur exactly once',used))
    nodes=[]
    def collect(node):
        nodes.append(node)
        for child in node.children:collect(child)
    collect(oldroot)
    adjacency={node:[] for node in nodes}
    for node in nodes:
        for child in node.children:
            adjacency[node].append(child);adjacency[child].append(node)
    left,right=oldroot.children
    adjacency[left].remove(oldroot);adjacency[right].remove(oldroot)
    adjacency[left].append(right);adjacency[right].append(left)
    del adjacency[oldroot]
    root=next(node for node in nodes if not node.children)
    for node in nodes:node.children=[];node.parent=None
    def orient(node,parent=None):
        for child in adjacency[node]:
            if child is parent:continue
            node.add_child(child);orient(child,node)
    orient(root)
    tree=BasisTree(root)
    tree.central_node_index=tree.node_idx[right]
    tree.hierarchy=hierarchy;tree.topology_label=label
    assert len(tree.node_list)==2*len(leaves)-2
    assert len(tree.root.children)==1
    assert all(len(n.children) in (0,2) for n in tree.node_list[1:])
    return tree

def balanced_tree(leaves,qn_size=1):
    leaves=list(leaves)
    return hierarchy_tree(leaves,balanced_hierarchy(range(len(leaves))),qn_size,
                          label='balanced_by_leaf_count_validation_baseline')

def ir_tree(leaves):
    """CH3CN leaf-only hierarchy implemented in Experiment/utils_ch3cn.py.

    Zero-based normal-mode numbering. The cited local implementation follows
    Larsson's optimized CH3CN topology, also used by Sun et al., Fig. 9(c).
    This is a reproducible baseline, not a claim of optimal IR-response cost.
    """
    leaves=list(leaves)
    if len(leaves)!=12:raise ValueError('CH3CN requires 12 ordered modes')
    return hierarchy_tree(leaves,IR_HIERARCHY,label='CH3CN_literature_leaf_only')

def aggregate_tree(leaves,molecules,modes_per_molecule,local_order='electron_fast'):
    """Candidate: preserve molecular blocks before grouping adjacent molecules.

    Each leaf already contains its physical/auxiliary pair. A slow-mode model
    admits two local binary resolutions; their accuracy must be compared.
    This Hamiltonian-guided candidate is not a topology from Jiang's MPS paper.
    """
    leaves=list(leaves);width=modes_per_molecule+1
    if modes_per_molecule not in (1,2) or len(leaves)!=molecules*width:
        raise ValueError('Expected electronic and one/two phonon leaves per molecule')
    if local_order not in ('electron_fast','phonons'):
        raise ValueError(local_order)
    blocks=[]
    for i in range(molecules):
        e=width*i
        blocks.append((e,e+1) if width==2 else
                      (((e,e+1),e+2) if local_order=='electron_fast' else (e,(e+1,e+2))))
    tree=hierarchy_tree(leaves,balanced_hierarchy(blocks),qn_size=2,
                        label='molecular_blocks_'+local_order+'_candidate')
    tree.molecular_blocks=blocks
    return tree

def describe(tree,state=None):
    return [dict(node=i,parent=None if n.parent is None else tree.node_idx[n.parent],
        children=[tree.node_idx[c] for c in n.children],physical_dims=n.pbond_dims,
        dofs=[repr(b.dofs) for b in n.basis_sets],
        bond=None if state is None else state.node_list[i].shape[-1],
        central_edge=i==getattr(tree,'central_node_index',None)) for i,n in enumerate(tree)]
