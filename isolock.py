#!/usr/bin/python
import copy
import os
import sys
import numpy as np
import re
import networkx as nx
import random
import time
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

def ExtractSingleMultiOutputNodes(G):
    F_multi=[]
    F_single=[]
    for n in G.nodes():
        out_degree=G.out_degree(n)
        check=G.nodes[n]['output']
        if out_degree == 1:
            if G.nodes[n]['gate'] != "input" and not check:
                F_single.append(n)
        elif out_degree>1:
            if G.nodes[n]['gate'] != "input" and not check:
                F_multi.append(n)
        else:
            if not check:
                print("Node "+n+" has 0 output and it is not an output")
    return F_multi, F_single

def verilog2gates(verilog, dump):
    Dict_gates={'xor':[0,1,0,0,0,0,0,0],
    'XOR':[0,1,0,0,0,0,0,0],
    'OR':[0,0,1,0,0,0,0,0],
    'or':[0,0,1,0,0,0,0,0],
    'XNOR':[0,0,0,1,0,0,0,0],
    'xnor':[0,0,0,1,0,0,0,0],
    'and':[0,0,0,0,1,0,0,0],
    'AND':[0,0,0,0,1,0,0,0],
    'nand':[0,0,0,0,0,1,0,0],
    'buf':[0,0,0,0,0,0,0,1],
    'BUF':[0,0,0,0,0,0,0,1],
    'NAND':[0,0,0,0,0,1,0,0],
    'not':[0,0,0,0,0,0,1,0],
    'NOT':[0,0,0,0,0,0,1,0],
    'nor':[1,0,0,0,0,0,0,0],
    'NOR':[1,0,0,0,0,0,0,0],
}
    G = nx.DiGraph()
    ML_count=0
    regex= "\s*(\S+)\s*=\s*(BUF|NOT)\((\S+)\)\s*"
    for output, function, net_str in re.findall(regex,verilog,flags=re.I |re.DOTALL):
        input=net_str.replace(" ","")


        G.add_edge(input,output)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:

            feat=Dict_gates[function]
            for item in feat:
                f_feat.write(str(item)+" ")
            f_feat.write("\n")
            f_cell.write(str(ML_count)+" assign for output "+output+"\n")
            f_count.write(str(ML_count)+"\n")
        ML_count+=1
    regex= "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex,verilog,flags=re.I | re.DOTALL):
        nets = net_str.replace(" ","").replace("\n","").replace("\t","").split(",")
        inputs = nets
        G.add_edges_from((net,output) for net in inputs)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:
            feat=Dict_gates[function]
            for item in feat:
                f_feat.write(str(item)+" ")
            f_feat.write("\n")
            f_cell.write(str(ML_count)+" assign for output "+output+"\n")
            f_count.write(str(ML_count)+"\n")
        ML_count+=1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    for n in G.nodes:
        G.nodes[n]['output'] = False
    out_regex = "OUTPUT\((.+?)\)\n"
    for net_str in re.findall(out_regex,verilog,flags= re.I | re.DOTALL):
        nets = net_str.replace(" ","").replace("\n","").replace("\t","").split(",")
        for net in nets:
            if net not in G:
                print("Output "+net+" is Float")
            else:
                G.nodes[net]['output'] = True
    if dump:
        for u,v in G.edges:
            if 'count' in G.nodes[u].keys() and 'count' in G.nodes[v].keys():
                f_link_train.write(str(G.nodes[u]['count'])+" "+str(G.nodes[v]['count'])+"\n")
    return G

# ----------------- IsoLock Strategy ----------------- #
def is_in_out_cone(G, u, v):
    return nx.has_path(G, u, v) or nx.has_path(G, v, u)


def iso_check_fast(G1, G2):
    # Relaxed Isocheck: check for gate set similarity
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False
    g1_gates = [G1.nodes[n].get('gate', '') for n in G1]
    g2_gates = [G2.nodes[n].get('gate', '') for n in G2]
    g1_count = defaultdict(int)
    g2_count = defaultdict(int)
    for g in g1_gates:
        g1_count[g] += 1
    for g in g2_gates:
        g2_count[g] += 1
    common = sum(min(g1_count[k], g2_count[k]) for k in g1_count)
    similarity = common / len(g1_gates) if g1_gates else 0
    return similarity >= 0.8


def iso_check_vf1(G1, G2):
    """ IsoCheck Variant Function I """
    # Check if number of nodes is equal
    if G1.order() != G2.order():
        return False
    # Extract and sort node degrees
    d1 = sorted(d for _, d in G1.degree())
    d2 = sorted(d for _, d in G2.degree())
    # Compare sorted degree lists
    return d1 == d2


def iso_check_vf2(G1, G2):
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

# def find_isomorphism(Si, Fsingle, Fmulti, Imax, Omax, G, h):
def find_isomorphism(Si, Fsingle, Fmulti, Imax, Omax, G, h, reachable_map):
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
            # if g1 == g2 or is_in_out_cone(G, f2, g1) or is_in_out_cone(G, f1, g2): continue   # this condition will greatly decrease possible solutions, but result a more strict condition which avoid the loops
            if g1 == g2 or g1 in reachable_map.get(f2, set()) or g2 in reachable_map.get(f1, set()): continue   # this is a faster implement of the above one
            sub_f1g1 = get_subgraph_between(G, f1, g1, h)
            sub_f2g1 = get_subgraph_between(G, f2, g1, h)
            sub_f1g2 = get_subgraph_between(G, f1, g2, h)
            sub_f2g2 = get_subgraph_between(G, f2, g2, h)
            if all(iso_check_vf1(a, b) for a, b in [
                (sub_f1g1, sub_f2g1), (sub_f1g2, sub_f2g2), (sub_f1g1, sub_f1g2), (sub_f2g1, sub_f2g2)]):
                return [f1, f2], [g1, g2], True
            # print(f"Trying pair: ({f1}, {f2}) → ({g1}, {g2})")
            # print(f"sub_f1g1 nodes: {sub_f1g1.nodes()}")
            # print(f"sub_f2g1 nodes: {sub_f2g1.nodes()}")
            # print(f"iso_check_vf1: {iso_check_vf1(sub_f1g1, sub_f2g1)}")

    return [None, None], [None, None], False

def isolock_locking_scheme(G, K, Ls, Imax, Omax, h, limit=0, max_iter=None):
    start_time = time.time()
    Fsingle, Fmulti = [], []
    if max_iter is None:
        max_iter = 10 * len(K)
    num_locked_pairs = 0
    Fmulti, Fsingle = ExtractSingleMultiOutputNodes(G)
    reachable_map = {n: nx.descendants(G, n) for n in G.nodes()}    # Precompute reachability map to speed up has_path checks
    locked_pairs, attempts = [], 0
    while len(locked_pairs) < len(K):
        Ssel = random.choice(Ls)
        # (f1, f2), (g1, g2), done = find_isomorphism(Ssel, Fsingle, Fmulti, Imax, Omax, G, h)
        (f1, f2), (g1, g2), done = find_isomorphism(Ssel, Fsingle, Fmulti, Imax, Omax, G, h, reachable_map)
        if done:
            locked_pairs.append(((f1, f2), (g1, g2), K[len(locked_pairs)]))
            num_locked_pairs = num_locked_pairs + 1
            print(f"No. {num_locked_pairs} Key bits has been locked, {Ssel} method selected")
        attempts += 1
        if (attempts % 10 ==0):
            elapsed = time.time() - start_time
            print(f"{attempts} attempts made (elapsed: {elapsed:.2f}s)")
            # print(f"{num_locked_pairs} Key bits has been locked so far")
        if ((attempts == max_iter) and limit):
            print("Reached the max iterations")
            print(f"{len(K)-num_locked_pairs} Key bits remain")
            break
    return locked_pairs


def create_key_dependent_routing(G, iso_pairs, key_bits):
    """Create key-dependent routing between isomorphic pairs by inserting MUX and modifying graph structure"""
    locked_graph = copy.deepcopy(G)
    myDict = {}
    selected_gates = []

    for idx, ((f1, f2), (g1, g2)) in enumerate(iso_pairs):
        mux_node = f"{g1}_from_mux"
        locked_graph.add_node(mux_node, gate='MUX', count=locked_graph.number_of_nodes())

        # Remove original edges to g1
        if locked_graph.has_edge(f1, g1):
            locked_graph.remove_edge(f1, g1)
        if locked_graph.has_edge(f2, g1):
            locked_graph.remove_edge(f2, g1)

        # Add edges to MUX node
        locked_graph.add_edge(f1, mux_node)
        locked_graph.add_edge(f2, mux_node)

        # Redirect g1's outgoing edges to mux output
        for succ in list(locked_graph.successors(g1)):
            locked_graph.add_edge(mux_node, succ)
            locked_graph.remove_edge(g1, succ)

        myDict[g1] = (f1, f2, idx)
        selected_gates.append(g1)

    return locked_graph, myDict, selected_gates


def generate_locked_circuit(G, iso_pairs, key_bits, benchmark, location):
    locked_graph, myDict, selected_gates = create_key_dependent_routing(G, iso_pairs, key_bits)

    with open(f"{location}/{benchmark}_K{len(key_bits)}.bench", "w") as locked_file:
        locked_file.write("#key=" + "".join(str(int(b)) for b in key_bits) + "\n")
        for i in range(len(key_bits)):
            locked_file.write(f"INPUT(keyinput{i})\n")

        with open(f"../Benchmarks_Original/{benchmark}.bench", 'r') as file1:
            lines = file1.readlines()

        for line in lines:
            line = line.strip()
            modified = False

            if any(ext + " =" in line for ext in selected_gates):
                regex = r"(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
                for output, function, net_str in re.findall(regex, line, flags=re.I | re.DOTALL):
                    if output in myDict:
                        f1, f2, key_idx = myDict[output]
                        line = line.replace(f1 + ",", output + "_from_mux,")
                        line = line.replace(f1 + ")", output + "_from_mux)")
                        locked_file.write(line + "\n")
                        locked_file.write(
                            f"{output}_from_mux = MUX(keyinput{key_idx}, {f1}, {f2})\n"
                        )
                        modified = True
                if not modified:
                    locked_file.write(line + "\n")
            else:
                locked_file.write(line + "\n")

    return locked_graph, myDict, selected_gates

# ----------------- Main Func ----------------- #
if __name__ == "__main__":
    total_start_time = time.time()
    G = parse(f"../Benchmarks_Original/{benchmark}.bench")
    K = GenerateKey(key_size)
    strategy_list = ["S1", "S2", "S3", "S4"]
    print("Running IsoLock strategy-based isomorphism search...")
    locked_pairs = isolock_locking_scheme(G, K, strategy_list, Imax=100, Omax=50, h=2, limit=0)

    iso_pairs = [(list(pair[0]), list(pair[1])) for pair in locked_pairs if None not in pair[0] and None not in pair[1]]
    if len(iso_pairs) < key_size:
        print(f"Warning: Only found {len(iso_pairs)} valid pairs, trimming key")
        K = K[:len(iso_pairs)]

    print("Generating locked circuit with MUX logic and ML data...")
    # locked_graph, routing_info = generate_locked_circuit(G, iso_pairs, K, benchmark, location)
    locked_graph, myDict, selected_gates = generate_locked_circuit(G, iso_pairs, K, benchmark, location)
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
    
    total_elapsed = time.time() - total_start_time
    with open(os.path.join(location, "TimeSpent.txt"), "w") as tf:
        tf.write(f"Total time spent: {total_elapsed:.2f} seconds\n")
