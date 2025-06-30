#!/usr/bin/python
import copy
import os
import sys
import numpy as np
import re
import networkx as nx
import random
from itertools import combinations, permutations
from collections import defaultdict

benchmark = sys.argv[1]
key_size = int(sys.argv[2])
location = sys.argv[3]
print("Benchmark is " + benchmark)
print("Key-size is " + str(key_size))
os.makedirs(location, exist_ok=True)

# ----------------- Utility and Parsing ----------------- #
def GenerateKey(K):
    nums = np.ones(K)
    nums[0:(K//2)] = 0
    np.random.shuffle(nums)
    return nums

def parse(path, dump=False):
    top = path.split('/')[-1].replace('.bench', '')
    with open(path, 'r') as f:
        data = f.read()
    return verilog2gates(data, dump)

def verilog2gates(verilog, dump):
    Dict_gates = {
        'xor': [0,1,0,0,0,0,0,0], 'XOR': [0,1,0,0,0,0,0,0],
        'OR': [0,0,1,0,0,0,0,0], 'or': [0,0,1,0,0,0,0,0],
        'XNOR': [0,0,0,1,0,0,0,0], 'xnor': [0,0,0,1,0,0,0,0],
        'and': [0,0,0,0,1,0,0,0], 'AND': [0,0,0,0,1,0,0,0],
        'nand': [0,0,0,0,0,1,0,0], 'NAND': [0,0,0,0,0,1,0,0],
        'buf': [0,0,0,0,0,0,0,1], 'BUF': [0,0,0,0,0,0,0,1],
        'not': [0,0,0,0,0,0,1,0], 'NOT': [0,0,0,0,0,0,1,0],
        'nor': [1,0,0,0,0,0,0,0], 'NOR': [1,0,0,0,0,0,0,0],
        'MUX': [0,0,0,0,0,0,0,1],
    }
    G = nx.DiGraph()
    ML_count = 0
    regex1 = r"(\S+)\s*=\s*(BUF|NOT)\((\S+)\)"
    for out, func, inp in re.findall(regex1, verilog, flags=re.I):
        G.add_edge(inp, out)
        G.nodes[out]['gate'] = func.upper()
        G.nodes[out]['count'] = ML_count
        ML_count += 1
    regex2 = r"(\S+)\s*=\s*(AND|OR|XOR|NAND|NOR|XNOR)\((.+?)\)"
    for out, func, net_str in re.findall(regex2, verilog, flags=re.I):
        inputs = net_str.replace(" ", "").split(",")
        for i in inputs:
            G.add_edge(i, out)
        G.nodes[out]['gate'] = func.upper()
        G.nodes[out]['count'] = ML_count
        ML_count += 1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
        G.nodes[n]['output'] = False
    out_regex = r"OUTPUT\((.+?)\)"
    for net_str in re.findall(out_regex, verilog):
        for net in net_str.replace(" ", "").split(","):
            if net in G:
                G.nodes[net]['output'] = True
    return G

# ----------------- IsoLock Strategy ----------------- #
def is_in_out_cone(G, u, v):
    return nx.has_path(G, u, v) or nx.has_path(G, v, u)

def iso_check(G1, G2):
    """ IsoCheck Variant Function II """
    if G1.order() != G2.order(): return False
    d1, d2 = dict(G1.degree()), dict(G2.degree())
    t1, t2 = nx.triangles(G1.to_undirected()), nx.triangles(G2.to_undirected())
    props1 = sorted([[d1[v], t1[v]] for v in d1])
    props2 = sorted([[d2[v], t2[v]] for v in d2])
    return props1 == props2

def get_nodes_within_hops(G, node, hop):
    nodes, frontier = set([node]), set([node])
    for _ in range(hop):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        frontier = next_frontier - nodes
        nodes.update(frontier)
    return nodes

def get_subgraph_between(G, node1, node2, hop):
    nodes1 = get_nodes_within_hops(G, node1, hop)
    nodes2 = get_nodes_within_hops(G, node2, hop)
    return G.subgraph(nodes1.union(nodes2)).copy()

def find_isomorphism(Si, Fsingle, Fmulti, Imax, Omax, G, h):
    F1, F2 = set(), set()
    if Si == "S3": F1, F2 = set(Fmulti), set(Fsingle)
    elif Si in ("S1", "S2"): F1, F2 = set(Fmulti), set(Fmulti)
    else: F1 = F2 = (set(Fsingle + Fmulti))
    for _ in range(Imax):   # Imax set the max iterations of Inputs selection (f1+f2)
        if not F1 or not F2: 
            print("F1 or F2 is empty!!")
            break
        f1, f2 = random.choice(list(F1)), random.choice(list(F2))
        while f1 == f2 and len(F2) > 1:
            f2 = random.choice(list(F2))
        for _ in range(Omax):   # Omax set the max iterations of Output selection (g1+g2)
            s1, s2 = list(G.successors(f1)), list(G.successors(f2))
            if not s1 or not s2: continue
            g1, g2 = random.choice(s1), random.choice(s2)
            if g1 == g2 or is_in_out_cone(G, f2, g1) or is_in_out_cone(G, f1, g2): continue   # this condition will greatly decrease possible solutions, but result a more strict condition which avoid the loops
            sub_f1g1 = get_subgraph_between(G, f1, g1, h)
            sub_f2g1 = get_subgraph_between(G, f2, g1, h)
            sub_f1g2 = get_subgraph_between(G, f1, g2, h)
            sub_f2g2 = get_subgraph_between(G, f2, g2, h)
            if all(iso_check(a, b) for a, b in [
                (sub_f1g1, sub_f2g1), (sub_f1g2, sub_f2g2), (sub_f1g1, sub_f1g2), (sub_f2g1, sub_f2g2)]):
                return [f1, f2], [g1, g2], True
            # print(f"Trying pair: ({f1}, {f2}) → ({g1}, {g2})")
            # print(f"sub_f1g1 nodes: {sub_f1g1.nodes()}")
            # print(f"sub_f2g1 nodes: {sub_f2g1.nodes()}")
            # print(f"iso_check: {iso_check(sub_f1g1, sub_f2g1)}")

    return [None, None], [None, None], False

def isolock_locking_scheme(G, K, Ls, Imax, Omax, h):
    Fsingle, Fmulti = [], []
    num_locked_pairs = 0
    for v in G.nodes():
        out_deg = len(list(G.successors(v)))
        if out_deg > 1: Fmulti.append(v)
        elif out_deg == 1: Fsingle.append(v)
    locked_pairs, attempts = [], 0
    while len(locked_pairs) < len(K) and attempts < 10 * len(K):
        Ssel = random.choice(Ls)
        (f1, f2), (g1, g2), done = find_isomorphism(Ssel, Fsingle, Fmulti, Imax, Omax, G, h)
        if done:
            locked_pairs.append(((f1, f2), (g1, g2), K[len(locked_pairs)]))
            num_locked_pairs = num_locked_pairs + 1
            print(f"No. {num_locked_pairs} Key bits has been locked, {Ssel} method selected")
        attempts += 1
        if (attempts % 10 ==0):
            print(f"{attempts} attempts made")
            print(f"{num_locked_pairs} Key bits has been locked so far")
    return locked_pairs

# ----------------- Main Func ----------------- #
if __name__ == "__main__":
    G = parse(f"./Benchmarks/{benchmark}.bench")
    K = GenerateKey(key_size)
    strategy_list = ["S1", "S2", "S3", "S4"]
    print("Running IsoLock strategy-based isomorphism search...")
    locked_pairs = isolock_locking_scheme(G, K, strategy_list, Imax=100, Omax=50, h=3)

    iso_pairs = [(list(pair[0]), list(pair[1])) for pair in locked_pairs if None not in pair[0] and None not in pair[1]]
    if len(iso_pairs) < key_size:
        print(f"Warning: Only found {len(iso_pairs)} valid pairs, trimming key")
        K = K[:len(iso_pairs)]

    print("Generating locked circuit with MUX logic and ML data...")
    locked_graph = copy.deepcopy(G)
    routing_info = {}
    Dict_gates = {
        'xor': [0,1,0,0,0,0,0,0], 'XOR': [0,1,0,0,0,0,0,0],
        'OR': [0,0,1,0,0,0,0,0], 'or': [0,0,1,0,0,0,0,0],
        'XNOR': [0,0,0,1,0,0,0,0], 'xnor': [0,0,0,1,0,0,0,0],
        'and': [0,0,0,0,1,0,0,0], 'AND': [0,0,0,0,1,0,0,0],
        'nand': [0,0,0,0,0,1,0,0], 'NAND': [0,0,0,0,0,1,0,0],
        'buf': [0,0,0,0,0,0,0,1], 'BUF': [0,0,0,0,0,0,0,1],
        'not': [0,0,0,0,0,0,1,0], 'NOT': [0,0,0,0,0,0,1,0],
        'nor': [1,0,0,0,0,0,0,0], 'NOR': [1,0,0,0,0,0,0,0],
        'MUX': [0,0,0,0,0,0,0,1],
    }
    ML_count = 0
    f_feat = open(location + "/feat.txt", "w")
    f_cell = open(location + "/cell.txt", "w")
    f_count = open(location + "/count.txt", "w")
    f_link_test = open(location + "/links_test.txt", "w")
    f_link_train = open(location + "/links_train.txt", "w")
    f_link_test_neg = open(location + "/link_test_n.txt", "w")

    for n in locked_graph.nodes():
        if 'gate' in locked_graph.nodes[n] and locked_graph.nodes[n]['gate'] != 'input':
            gate_type = locked_graph.nodes[n]['gate']
            if gate_type in Dict_gates:
                feat = Dict_gates[gate_type]
                for item in feat:
                    f_feat.write(str(item) + " ")
                f_feat.write("\n")
                f_cell.write(str(ML_count) + " assign for output " + str(n) + "\n")
                f_count.write(str(ML_count) + "\n")
                locked_graph.nodes[n]['count'] = ML_count
                ML_count += 1

    for u, v in G.edges():
        if 'count' in G.nodes[u] and 'count' in G.nodes[v]:
            f_link_train.write(str(G.nodes[u]['count']) + " " + str(G.nodes[v]['count']) + "\n")

    for idx, (nodes1, nodes2) in enumerate(iso_pairs):
        if nodes1 and nodes2:
            n1, n2 = random.choice(nodes1), random.choice(nodes2)
            if 'count' in locked_graph.nodes[n1] and 'count' in locked_graph.nodes[n2]:
                f_link_test.write(str(locked_graph.nodes[n1]['count']) + " " + str(locked_graph.nodes[n2]['count']) + "\n")
                f_link_test_neg.write(str(locked_graph.nodes[n2]['count']) + " " + str(locked_graph.nodes[n1]['count']) + "\n")

    f_feat.close()
    f_cell.close()
    f_count.close()
    f_link_test.close()
    f_link_train.close()
    f_link_test_neg.close()
    print(f"Locked circuit and ML data written to {location}/")
