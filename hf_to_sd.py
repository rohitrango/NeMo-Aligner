from typing import Any
import torch
import numpy as np
import json
from pprint import pprint
from compare_sdxl_keys import *

vt_fn = lambda x: np.sum([np.prod(list(t)) for t in x])

def split_name(name, dots=0):
    l = name.split(".")
    return ".".join(l[:dots+1]), ".".join(l[dots+1:])

def is_prefix(shortstr, longstr):
    # is the first string a prefix of the second one
    return longstr.startswith(shortstr)

def numdots(str):
    return str.count(".")

class SegTree:
    def __init__(self):
        self.nodes = dict()
        self.val = None
        self.final_val = 0
    
    def add(self, name, val=0):
        prefix, subname = split_name(name)
        if subname == '':
            self.nodes[name] = SegTree()
            self.nodes[name].val = val
            return
        if self.nodes.get(prefix) is None:
            self.nodes[prefix] = SegTree()
        self.nodes[prefix].add(subname, val)
    
    def change(self, name, val):
        self.add(name, val)
    
    def __getitem__(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        # val = self.nodes.get(name)
        val = self.nodes.get(name)
        if val is None:
            # straight lookup failed, do a prefix lookup
            keys = list(self.nodes.keys())
            p_flag = [is_prefix(k, name) for k in keys]
            if not any(p_flag):
                return None
            # either more than 1 match (error) or exactly 1 (success)
            if np.sum(p_flag) > 1:
                print(f"error: multiple matches of key {name} with {keys}")
            else:
                i = np.where(p_flag)[0][0]
                n = numdots(keys[i])
                prefix, substr = split_name(name, n)
                return self.nodes[prefix][substr]
            # start, subseq = split_name(name)
            # if subseq == "":
            #     return self.nodes.get(start)
            # else:
            #     return (self.nodes[start])[subseq]
        return val

def strip_wb_keys(keys):
    # strip the weight and bias
    newkeys = [x.replace(".weight", "").replace(".bias", "") for x in keys]
    newkeys = list(set(newkeys))
    return newkeys


def traverse_segtree(tree, tab=0, max_depth=np.inf, v_transform_fn=lambda x: x):
    if tab >= max_depth:
        return
    tabstr = "  "*tab
    for k, v in tree.nodes.items():
        print(f"{tabstr} {k}:")
        if tab < max_depth:
            traverse_segtree(v, tab+1, max_depth, v_transform_fn)
    # if no children, then this is a child node
    if len(tree.nodes) == 0:
        print(f"{tabstr} {v_transform_fn(tree.val)}")


def stat_segtree(tree, tab=0, max_depth=np.inf, \
                v_transform_fn=vt_fn,
                agg_fn = lambda x: sum(x)):

    tabstr = "  "*tab
    final_vals = []
    if tab >= max_depth:  # max depth reached, simply aggregate results 
        # for k, v in tree.nodes.items():
        keylist = sorted(list(tree.nodes.keys()))
        if len(keylist) > 0:
            for k in keylist:
                v = tree[k]
                final_vals.append(stat_segtree(v, tab+1, max_depth, v_transform_fn, agg_fn))
        else:
            final_vals.append(v_transform_fn(tree.val))
    else:
        # aggregate results
        # for k, v in tree.nodes.items():
        keylist = sorted(list(tree.nodes.keys()))
        if len(keylist) > 0:
            for k in keylist:
                v = tree[k]
                s = stat_segtree(v, tab+1, max_depth, v_transform_fn, agg_fn)
                print(f"{tabstr} - {k}: {s}")
                final_vals.append(s)
        else:
            val = v_transform_fn(tree.val)
            print(f"{tabstr} : {val}")
            final_vals.append(val)
    # save this in segtree
    final_v = agg_fn(final_vals)
    tree.final_val = final_v
    return final_v

def combine_subtrees(tree: SegTree, keys):
    # `keys` are list of node keys in tree
    newtree = SegTree()
    for key in keys:
        subtree = tree.nodes[key]
        if len(subtree.nodes) == 0: newtree.nodes[key] = subtree
        else:
            for k, v in subtree.nodes.items():
                newtree.nodes[key + "/" + k] = v
        newtree.final_val += subtree.final_val
    return newtree

# collapse segtree 
def collapse_tree(tree: SegTree, delim="."):
    # the idea is that if a node has only one child, then inherit that child's attributes, and rename the dict item
    nodes = tree.nodes
    if len(nodes) == 0:  # leaf node, do nothing
        return tree
    else:
        # collapse subtrees
        for k, subtree in tree.nodes.items():
            tree.nodes[k] = collapse_tree(subtree)
        # if only one child, collapse
        if len(nodes) == 1:
            newnodes = {}
            for prefix, subtree in nodes.items():   # this will only be 1 item
                for k, subsubtree in subtree.nodes.items():
                    newnodes[prefix + delim + k] = subsubtree
                print(f"Collapsed {k} into {prefix}{delim}{k}")
            tree.nodes = newnodes
            tree.val = subtree.val
            tree.final_val = subtree.final_val
        return tree
        

def isleaf(tree: SegTree):
    return len(tree.nodes) == 0

def remove_from_set(set, val):
    if val in set:
        set.remove(val)

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

nemo_path = "/opt/nemo-aligner/examples/mm/stable_diffusion/diffusion_model.ckpt"
hf_unet_path = "/opt/nemo-aligner/checkpoints/sdxl/unet.ckpt"
hf_vae_path = "/opt/nemo-aligner/checkpoints/sdxl/vae.bin"

nemo_data = torch.load(nemo_path) 

## get unet
nemo_unet = filter_keys(lambda key: key.startswith("model.diffusion_model"), nemo_data)
nemo_unet = map_keys(lambda key: key.replace("model.diffusion_model.", ""), nemo_unet)
hf_unet = torch.load(hf_unet_path)

def model_to_tree(model):
    keys = model.keys()
    keys = strip_wb_keys(keys)
    tree = SegTree()
    for k in keys:
        wk, bk = k + '.weight', k + '.bias' 
        wk = model.get(wk, torch.tensor([])).shape
        bk = model.get(bk, torch.tensor([])).shape
        tree.add(k, (wk, bk))
    return tree

hf_tree = model_to_tree(hf_unet)
nemo_tree = model_to_tree(nemo_unet)

# hf_tree = collapse_tree(hf_tree)
# nemo_tree = collapse_tree(nemo_tree)

stat_segtree(nemo_tree, v_transform_fn=vt_fn)
stat_segtree(hf_tree, v_transform_fn=vt_fn)

def combine_trees(*treenames):
    newtree = SegTree()
    for tree, name in treenames:
        # print(tree.nodes, name)
        if len(tree.nodes) > 0:
            for k, v in tree.nodes.items():
                newtree.nodes[name + "/" + k] = v
        else:
            # leaf node
            newtree.nodes[name] = tree
        newtree.final_val += tree.final_val
    return newtree

# upto this part it works fantastically, now try the hard part (matching)
def try_graph_matching(tree1, tree2, tab=0, key1=None, key2=None, combine_hints=combine_hint_list):
    '''
    key logic: map the parts that are equal first

    check for keys that are unmatched. for the ones unmatched, we will map them later.
    if there are multiple matches but no unmatched, then map them using some heuristic
    '''
    if key1 is not None:
        return try_graph_matching(tree1[key1], tree2, tab, None, key2)
    elif key2 is not None:
        return try_graph_matching(tree1, tree2[key2], tab, key1, None)

    if len(tree1.nodes) == 1:
        try_graph_matching(list(tree1.nodes.values())[0], tree2, tab)
        return
    elif len(tree2.nodes) == 1:
        try_graph_matching(tree1, list(tree2.nodes.values())[0], tab)
        return
    
    # keep track of unmatched
    unmatched_1 = set(list(tree1.nodes.keys()))
    unmatched_2 = set(list(tree2.nodes.keys()))

    tabstr = "\t"*tab
    nodes1 = {k: tree1.nodes[k].final_val for k in tree1.nodes.keys()}
    node1match = {k: [] for k in nodes1}
    nodes2 = {k: tree2.nodes[k].final_val for k in tree2.nodes.keys()}
    # find the equal ones
    for k1, v1 in nodes1.items():
        for k2, v2 in nodes2.items():
            if v1 == v2:
                node1match[k1].append(k2)
                remove_from_set(unmatched_1, k1)
                remove_from_set(unmatched_2, k2)
    # print exact matches 
    nodematchkeys = list(node1match.keys())
    # multiple matches
    multimatch = {}
    for k1 in nodematchkeys:
        k2s = node1match[k1]
        # this is a case of exact match found, no further investigation
        if len(k2s) == 1:
            l1, l2 = isleaf(tree1.nodes[k1]), isleaf(tree2.nodes[k2s[0]])
            if l1 and l2:
                leafmatch = "leaf"
            elif not l1 and not l2:
                leafmatch = "node"
            else:
                leafmatch = f"mismatch {l1} {l2}"

            print(f"{tabstr}Exact match found between {k1} and {k2s[0]} ({leafmatch})")
            try_graph_matching(tree1[k1], tree2[k2s[0]], tab+1)
            del nodes1[k1], node1match[k1]
            del nodes2[k2s[0]]
        elif len(k2s) > 1:
            # print(f"{tabstr}Multiple exact matches found. {k1} {k2s}.")
            multimatch[k1] = sorted(k2s)
    
    # if multi = True, use some heuristic
    if len(multimatch) > 0:
        k1 = len(multimatch)
        k2l = [len(multimatch[x]) for x in multimatch.keys()]
        if all([k1 == x for x in k2l]):
            # apply some heuristic here
            k1sorted = sorted(list(multimatch.keys()))
            for i, k1 in enumerate(k1sorted):
                k2m = multimatch[k1][i]
                print(f"{tabstr} Trying matching {k1} with {k2m}...")
                try_graph_matching(tree1[k1], tree2[k2m], tab+1)
        else:
            print("mismatch in number of multimatches.")
    
    # print unmatched keys
    if len(unmatched_1) > 0:
        print(f"{tabstr}Unmatched keys 1: {unmatched_1}")
    if len(unmatched_2) > 0:
        print(f"{tabstr}Unmatched keys 2: {unmatched_2}")

    ### try a bunch of hints here
    if len(unmatched_1) > 0 and len(unmatched_2) > 0:
        for t1keys, t2keys in combine_hints:
            if not isinstance(t1keys, (list, tuple, set)): t1keys = [t1keys]
            if not isinstance(t2keys, (list, tuple, set)): t2keys = [t2keys]
            if all([x in unmatched_1 for x in t1keys]) and all([x in unmatched_2 for x in t2keys]):
                # print(t1keys, t2keys, "trying hint..")
                # newtree1 = combine_trees(*[(tree1[k1], k1) for k1 in t1keys])
                # newtree2 = combine_trees(*[(tree2[k2], k2) for k2 in t2keys])
                newtree1 = combine_subtrees(tree1, t1keys)
                newtree2 = combine_subtrees(tree2, t2keys)
                try_graph_matching(newtree1, newtree2, tab+1)
                print()
    return

combine_hint_list = [
    (("conv_in", "down_blocks"), "input_blocks"),
    (("conv_norm_out", "conv_out"), "out"),
    (('down_blocks/0',), ("input_blocks/3", "input_blocks/1", "input_blocks/2")),
    (('down_blocks/1',), ("input_blocks/4", "input_blocks/5", "input_blocks/6")),
    (('down_blocks/2',), ("input_blocks/7", "input_blocks/8")),
]
    # ({'down_blocks/2', 'down_blocks/1', 'down_blocks/0'}, {'input_blocks/2', 'input_blocks/3', 'input_blocks/7', 'input_blocks/6', 'input_blocks/8', 'input_blocks/1', 'input_blocks/5', 'input_blocks/4'})
    

# def create_graph(state_dict):
#     ''' few assumptions 
#     - all keys will end in weight/bias, this gives a neat way to pair matching keys
#     - for a list of modules (numbered), just arrange them in order
#     - for a list of keys, arrange them in a dict
#     '''
#     keys = list(state_dict.keys())
#     graph = create_graph_helper(list(zip(keys, keys)), state_dict)
#     return graph

# stripheader = lambda x: ".".join(x.split(".")[1:])

# class LeafNode:
#     def __init__(self, shapes, keys=None, prefix=""):
#         # keys refers to fullkeys here
#         self.nodetype = "node"
#         self.prefix = prefix
#         self.shapes = shapes
#         self.n = np.sum([np.prod(list(x)) for x in shapes])
#         self.keys = keys

# class NonLeafNode:
#     def __init__(self, elements, prefix='/'):
#         self.elements = elements
#         self.prefix = prefix
#         if isinstance(elements, dict):
#             self.nodetype = "list"
#             self.n = np.sum([g.n for g in elements.values()])
#         elif isinstance(elements, list):
#             self.nodetype = "dict"
#             self.n = np.sum([g.n for g in elements])

# def create_graph_helper(tuplekey, state_dict, prefix="/"):
#     ''' given list of keys, create subgraph (this is recursive) '''
#     # base case
#     keylist, fullkeys = [[x[i] for x in tuplekey] for i in range(2)]

#     if len(keylist[0].split(".")) == 1:
#         assert [k in ['bias', 'weight'] for k in keylist], keylist
#         return LeafNode([state_dict[k].shape for k in fullkeys], fullkeys, prefix)

#     # there is at least one delimiter here
#     cg = set([x.split(".")[0] for x in keylist])
#     try:
#         cg = [int(x) for x in cg]
#         isint = True
#     except:
#         isint = False

#     subgraphs = dict()
#     # create sublists
#     for subg in cg:
#         subtuple = list(filter(lambda x: x[0].startswith(str(subg) + "."), tuplekey))
#         # strip the first part
#         subtuple = list(map(lambda x: (stripheader(x[0]), x[1]), subtuple))
#         subgraph = create_graph_helper(subtuple, state_dict, subg)
#         subgraphs[subg] = subgraph
#     # 
#     if isint:
#         subgraphs = [subgraphs[g] for g in sorted(cg)]

#     return NonLeafNode(subgraphs, prefix=prefix)


# def traverse_graph(graph, tabs=0):
#     start = "   " * tabs
#     # try:
#     if graph.nodetype == "node":
#         print(start , f"[{graph.prefix}]" ,  " ".join(list(graph.keys)) ,graph.shapes , graph.n)
#     elif graph.nodetype == 'list':
#         print(start + f"{graph.prefix} --- ")
#         for el in graph.elements:
#             traverse_graph(el, tabs+1)
#             print()
#     elif graph.nodetype == 'dict':
#         print(start + f"{graph.prefix} <>")
#         for k, v in graph.elements.items():
#             print(start + k)
#             traverse_graph(v, tabs+1)
#             print()

# except Exception as e:
#     print(graph, e)


# populate the nemo tree
# nemo_tree = SegTree()
# nemo_keys = strip_wb_keys(nemo_unet.keys())
# for k in nemo_keys:
#     wk, bk = k + '.weight', k + '.bias'
#     wk = nemo_unet.get(wk, torch.tensor([])).shape
#     bk = nemo_unet.get(bk, torch.tensor([])).shape 
#     nemo_tree.add(k, (wk, bk))


# populate the HF tree
# hf_tree = SegTree()
# hf_keys = strip_wb_keys(hf_unet.keys())
# for k in hf_keys:
#     wk, bk = k + '.weight', k + '.bias'
#     wk = hf_unet.get(wk, torch.tensor([])).shape
#     bk = hf_unet.get(bk, torch.tensor([])).shape 
#     hf_tree.add(k, (wk, bk))