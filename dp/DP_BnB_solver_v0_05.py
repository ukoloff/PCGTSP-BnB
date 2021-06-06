import networkx as nx
import Instance_generator as ig
import time
import multiprocessing as mp
import pickle as pic
import gc
import psutil
import os
import functools as ft
import sys
import glob 

from collections import Counter
from salmanize import lower_bound, nc0

MAXINT = 100000000000000000000000000000
MAXTASKSPERWORKER = 1000

###### Memory usage util #############################################################################
##  ATTENTION: designed for  Linux, for another OS can possible be adapted

MEM_AMPL = 1.5
MEM_RESERVE = 0.7

def possible_workers_count():
    proc = psutil.Process(os.getpid())
    used_this_process = proc.memory_info()[0] * MEM_AMPL
    available = psutil.virtual_memory()[1] * MEM_RESERVE
    return max(1, int(available / used_this_process))



####### We assume that our precedence constraints are defined by a spanning tree rooted at V_1 #######

class State:
    def __init__(self,sigma,j,tilde_u,v, cost = MAXINT):
        self.sigma = sigma       # index set of visited clusters except the first one
        self.j = j               # index of the last cluster
        self.tilde_u = tilde_u   # last node in the last cluster
        self.v = v               # first node 
        self.predecessor_witness = ''    # predecessor state witness
        self.cost = cost
        self.LB = - MAXINT  

    def witness(self):
        return f'{self.sigma}, {self.j}, {self.tilde_u}, {self.v}'

       

def make_unique_list(lst):
    tuples=map(lambda item: tuple(sorted(item)),lst)
    cnt = Counter(tuples)
    unique_tuples = list(cnt.keys())
    return list(map(list,unique_tuples))


def compute_Bellman_cell(G, clusters, transitive_closure, lookup_table, filtered_prev_states, state):        
    best_cost = MAXINT
    prev_state = None

    
    for suggested_prev_state in filtered_prev_states:
        cost = suggested_prev_state.cost + G[suggested_prev_state.tilde_u][state.tilde_u]['weight']
        if cost < best_cost:
            best_cost = cost
            prev_state = suggested_prev_state


    if best_cost < MAXINT:
        state.cost = best_cost
        state.predecessor_witness = prev_state.witness()
        lookup_table[state.witness()] = state

    return best_cost

################ multiprocessing staff #########
###  
def worker_init(G, clusters, tree, lookup_table_filename, UB):
    global mp_G
    global mp_clusters
    global mp_tree
    global mp_lookup_table
    global mp_transitive_closure
    global mp_nc0
    global mp_UB
    global mp_succs

    mp_G, mp_clusters, mp_tree = G, clusters, tree
    mp_transitive_closure = nx.transitive_closure(tree)
    mp_UB = UB

    mp_succs={}

    c_keys=list(mp_clusters.keys())
    start_cluster_id = c_keys[0]
    mp_nc0 = nc0(mp_G, mp_clusters, mp_tree, start_cluster_id)
    # print(f'NC0=\n{mp_nc0.edges(data="weight")}')

    with open(lookup_table_filename, 'rb') as fin:
        mp_lookup_table = pic.load(fin)

def can_be_the_last_cluster(sigma,c_ind, transitive_closure):
    if c_ind not in mp_succs:
        succ = transitive_closure.successors(c_ind)
        mp_succs[c_ind] = succ
    else:
        succ = mp_succs[c_ind]

    return all(not s in sigma for s in succ)

def parallel(sigma):
    global mp_lookup_table

    result ={}
    capacity = 0
    c_keys=list(mp_clusters.keys())
    start_cluster_id = c_keys[0]
    cutoff = 0
   
    for ind_V_j in [ind for ind in sigma if can_be_the_last_cluster(sigma, ind, mp_transitive_closure)]:
        ###
        try:
            P2_cost = lower_bound(mp_nc0, [start_cluster_id] + sigma, ind_V_j, start_cluster_id)
        except:
            print(f'LB calculations fault, sigma={sigma}, start={start_cluster_id}, dest={ind_V_j}')
            exit(1)
        ###
        # print(f'S={[start_cluster_id]+sigma},\t org_cluster={start_cluster_id},\t dest_cluster={ind_V_j},\t  P2_cost={P2_cost}')
        truncated_sigma = [ind for ind in sigma if ind != ind_V_j]
        for v in mp_clusters[start_cluster_id]:
            filtered_prev_states = [s for s in mp_lookup_table.values() if truncated_sigma == s.sigma and v == s.v]
            for tilde_u in mp_clusters[ind_V_j]:
                state = State(sigma, ind_V_j, tilde_u, v)
                state.cost = compute_Bellman_cell(mp_G, mp_clusters, mp_transitive_closure, mp_lookup_table, filtered_prev_states, state)
                if state.cost < MAXINT:
                    state.LB = P2_cost + state.cost
                    if state.LB <= mp_UB:
                        result[state.witness()] = state
                        capacity += 1
                    else:
                        cutoff += 1
    return (capacity, result, cutoff)
###
##################################################

def do_prepare_layer(sigma, tree, V1, V1_succ):
    current_layer_chunk=[]
    l_sigma = list(sigma) 
    current_layer_chunk += [l_sigma + [succ] for succ in V1_succ if succ not in l_sigma]
    
    for ind in l_sigma:
        suppl = [l_sigma + [succ] for succ in tree.successors(ind) if succ not in l_sigma and nx.ancestors(tree,succ).issubset(l_sigma+[V1])]
        current_layer_chunk += suppl

    return current_layer_chunk

def prepare_layer(clusters,tree, previous_layer, layer_level):
    clust_keys=list(clusters.keys())
    V1 = clust_keys[0]
    V1_succ = list(tree.successors(V1))
    start_time = time.time()

    current_layer = []

    if not previous_layer:     # baseline case 
        current_layer = [[succ] for succ in V1_succ]
        return current_layer

    for sigma in previous_layer:
        current_layer += do_prepare_layer(sigma, tree, V1, V1_succ)
        
    current_layer = make_unique_list(current_layer)
    return current_layer


def compute_Bellman_layer(G, clusters,  layer_level, previous_layer, tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count, UB):
    c_keys=list(clusters.keys())
    start_time = time.time()

    layer = prepare_layer(clusters, tree, previous_layer, layer_level)
    print(f'layer {layer_level+1:03d} is prepared')

    lookup_table = {}

    capacity = 0
    cutoff = 0
    if layer_level < 1: # baseline case
        actual_workers_count = 1
        for sigma in layer:
            for ind_V_j in sigma:
                for v in clusters[c_keys[0]]:
                    for tilde_u in clusters[ind_V_j]:
                        if G.has_edge(v,tilde_u):             # checking if graph has edge
                            capacity += 1
                            state = State(sigma, ind_V_j, tilde_u, v)
                            state.cost = G[v][tilde_u]["weight"]
                            witness = state.witness()
                            lookup_table[witness] = state
                        
                
    else:

        actual_workers_count = min(workers_count, predicted_workers_count)
        with mp.Pool(actual_workers_count, worker_init, (G, clusters, tree, f'{lookup_table_name}{layer_level-1:03d}.dct', UB), maxtasksperchild=MAXTASKSPERWORKER) as pool:
            results = pool.map(parallel, layer)
            pool.close()
            pool.join()
            pool = None
            layer = None
 
            capacity = sum(map(lambda item: item[0],results))

            cutoff = sum(map(lambda item: item[2],results))
     
            def instead_of_lambda(acc, res):
                acc.update(res[1])
                return acc

            lookup_table = ft.reduce(instead_of_lambda, results, {})

    with open(f'{lookup_table_name}{layer_level:03d}.dct', 'wb') as fout:
        pic.dump(lookup_table,fout)
        fout.close()

    ####### to filter out the next layer ###################
    sigmas_from_lookup_table = list(set([tuple(s.sigma) for s in lookup_table.values()]))
    ########################################################
        
    predicted_workers_count = possible_workers_count() 

    if not keep_lookup_table:
        lookup_table = None    
    
    gc.collect()

    print(f'layer {layer_level+1:03d} of size {capacity:>10} is completed by {actual_workers_count} worker(s) at {time.time() - start_time:8.2f} sec')
    print(f'{cutoff} ({cutoff / (cutoff + capacity):.1%}) of branches were cut off')
    return predicted_workers_count, lookup_table, sigmas_from_lookup_table


def DP_solver_layered(G, clusters, tree, lookup_table_name, need_2_keep_layers, workers_count, UB = MAXINT):
    
    num_of_layers = len(clusters) - 1

    predicted_workers_count = possible_workers_count()
    previous_layer = []

    for layer_level in range(num_of_layers):
        keep_lookup_table = (layer_level >= num_of_layers - 1)
        predicted_workers_count, lookup_table, previous_layer = compute_Bellman_layer(G, clusters,  layer_level, previous_layer, 
            tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count, UB)
        
    OPT = MAXINT
    best_state = None

    leaves = [ind_V for ind_V in clusters if tree.out_degree(ind_V) == 0]

    clust_keys = list(clusters.keys())
    sigma = sorted(clust_keys[1:])
    ind_V_1 = clust_keys[0]

   
    for v in clusters[ind_V_1]:
        print(f'leaves: {leaves}')
        for ind in leaves:
            for tilde_u in clusters[ind]:
                pretender_state = State(sigma, ind, tilde_u, v)
                witness = pretender_state.witness()
                if witness in lookup_table:
                    cached_state = lookup_table[witness]
                    cost = cached_state.cost + G[cached_state.tilde_u][v]['weight']
                    if cost < OPT:
                        OPT = cost
                        best_state = cached_state

    if not best_state:
        print(f'Instance is infeasible')
        return None

    path = [best_state.v, best_state.tilde_u]
    current_state = best_state
    route = [ind_V_1, best_state.j]

    layer_level = len(clusters) - 3
    
    while current_state.predecessor_witness != '':
         with open(f'{lookup_table_name}{layer_level:03d}.dct', 'rb') as fin:
            layer_lookup_table = pic.load(fin)
            predecessor_witness = current_state.predecessor_witness
            if not current_state.predecessor_witness in layer_lookup_table:
                print(f'state {predecessor_witness} not found at layer {layer_level+1}')
                exit(0)
            else:
                predecessor_state = layer_lookup_table[predecessor_witness]
                path.append(predecessor_state.tilde_u)
                route.append(predecessor_state.j)
                current_state = predecessor_state
                layer_level -= 1

    path.append(best_state.v)
    route.append(ind_V_1)
    path.reverse()
    route.reverse()

    if not need_2_keep_layers:
        filelist = glob.glob(f'{lookup_table_name}*.dct', recursive = False)
        for ff in filelist:
            try:
                os.remove(ff)
            except OSError as mag:
                print(msg)

    
    return (OPT, route, path)


def get_path_length(path, graph):
    dist = [graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
    print(f'dist = {dist}')
    return sum(dist)
      

def visited_clusters(tour, clusters):
    def cluster(node):
        found = -1
        for k in clusters.keys():
            if node in clusters[k]:
                found = k
                break
        return found
    return list(map(cluster, tour))