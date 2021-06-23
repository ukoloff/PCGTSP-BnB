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

from collections import Counter

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

    def witness(self):
        return f'{self.sigma}, {self.j}, {self.tilde_u}, {self.v}'

       

def make_unique_list(lst):
    tuples=map(tuple, lst)
    cnt = Counter(tuples)
    unique_tuples = list(cnt.keys())
    return list(map(list,unique_tuples))


################ multiprocessing staff ############
### 
def layer_worker_init(tree, V1_succ):
    global mp_tree
    global mp_V1_succ
    
    mp_tree, mp_V1_succ = tree, V1_succ

    
def parallel_make_layer(sigma):
    global mp_tree
    global mp_V1_succ

    current_layer_chunk=[]

    l_sigma = list(sigma) 
    current_layer_chunk += [l_sigma + [succ] for succ in mp_V1_succ if succ not in l_sigma]
    
    for ind in l_sigma:
        suppl = [l_sigma + [succ] for succ in mp_tree.successors(ind) if succ not in l_sigma]
        suppl = list(map(sorted, suppl))
        current_layer_chunk += suppl

    return make_unique_list(current_layer_chunk)


###
###################################################

def add_layer(clusters,tree, previous_layer, layer_level, workers_count):
    clust_keys=list(clusters.keys())
    V1_succ = list(tree.successors(clust_keys[0]))
    start_time = time.time()

    if len(previous_layer) == 0:     # baseline case 
        current_layer = [[succ] for succ in V1_succ]
        return current_layer

    actual_workers_count = min(workers_count, possible_workers_count())
    pool = mp.Pool(actual_workers_count, layer_worker_init, (tree, V1_succ), maxtasksperchild=MAXTASKSPERWORKER)
    
    results = pool.map(parallel_make_layer, previous_layer)
    pool.close()
    pool.join()
    
    # print('Pool is joined')
    assert len(results) > 0, 'results cannot be empty'

    current_layer = [sigma for result in results for sigma in result]
    
    current_layer = make_unique_list(current_layer)
    # print('Duplicates are excluded')
    print(f'layer {layer_level+1:03d} of size {len(current_layer):>8} is prepared by {actual_workers_count} worker(s) at {time.time() - start_time:8.2f} sec')
    return current_layer

def make_layers(clusters,tree, lookup_table_name, workers_count):
    num_of_layers = len(clusters) - 1
    # layers=[]
    previous_layer = []
    for layer_level in range(num_of_layers):
        current_layer = add_layer(clusters, tree, previous_layer, layer_level, workers_count)
        # layers.append(current_layer)

        with open(f'{lookup_table_name}{layer_level:03d}.lyr', 'wb') as fout:
            pic.dump(current_layer,fout)
            fout.close()       
        previous_layer = current_layer

        gc.collect()
    # return layers

def can_be_the_last_cluster(sigma,c_ind, transitive_closure):
    # desc = nx.descendants(tree,c_ind)
    succ = transitive_closure.successors(c_ind)
    return all(not s in sigma for s in succ)

def compute_Bellman_cell(G, clusters, transitive_closure, lookup_table, state):
    witness = state.witness()
    if witness in lookup_table:
        cached_state = lookup_table[witness]
        return cached_state.cost
        
    best_cost = MAXINT
    prev_state = None

    truncated_sigma = [ind for ind in state.sigma if ind != state.j]

    assert len(truncated_sigma) > 0, 'truncated_sigma is empty'

    for ind_prev_cluster in [ind for ind in truncated_sigma if can_be_the_last_cluster(truncated_sigma,ind,transitive_closure)]:
        for tilde_u in clusters[ind_prev_cluster]:
            suggested_prev_state = State(truncated_sigma,ind_prev_cluster,tilde_u,state.v)
            witness = suggested_prev_state.witness()
            if witness in lookup_table:
                pretender_state = lookup_table[witness]
                cost = pretender_state.cost + G[tilde_u][state.tilde_u]['weight']
                if cost < best_cost:
                    best_cost = cost
                    prev_state = pretender_state

    if best_cost < MAXINT:
        state.cost = best_cost
        state.predecessor_witness = prev_state.witness()
        lookup_table[state.witness()] = state

    return best_cost

################ multiprocessing staff #########
###  
def worker_init(G, clusters, tree, lookup_table_filename):
    global mp_G
    global mp_clusters
    global mp_tree
    global mp_lookup_table
    global mp_transitive_closure

    mp_G, mp_clusters, mp_tree = G, clusters, tree
    mp_transitive_closure = nx.transitive_closure(tree)

    with open(lookup_table_filename, 'rb') as fin:
        mp_lookup_table = pic.load(fin)

def parallel(sigma):
    global mp_G
    global mp_clusters
    global mp_tree
    global mp_lookup_table

    result ={}
    capacity = 0
    c_keys=list(mp_clusters.keys())

    for ind_V_j in [ind for ind in sigma if can_be_the_last_cluster(sigma, ind, mp_transitive_closure)]:
        for v in mp_clusters[c_keys[0]]:
            for tilde_u in mp_clusters[ind_V_j]:
                state = State(sigma, ind_V_j, tilde_u, v)
                state.cost = compute_Bellman_cell(mp_G, mp_clusters, mp_transitive_closure, mp_lookup_table, state)
                if state.cost < MAXINT:
                    result[state.witness()] = state
                    capacity += 1
    return (capacity, result)
###
##################################################

def compute_Bellman_layer(G, clusters,  layer_level, tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count):
 
    with open(f'{lookup_table_name}{layer_level:03d}.lyr', 'rb') as fin:
        layer = pic.load(fin)

    c_keys=list(clusters.keys())
    start_time = time.time()

    lookup_table = {}

    capacity = 0
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
        with mp.Pool(actual_workers_count, worker_init, (G, clusters, tree, f'{lookup_table_name}{layer_level-1:03d}.dct'), maxtasksperchild=MAXTASKSPERWORKER) as pool:
            results = pool.map(parallel, layer)
            pool.close()
            pool.join()
            pool = None
            layer = None
 
            capacity = sum(map(lambda item: item[0],results))
     
            def instead_of_lambda(acc, res):
                acc.update(res[1])
                return acc

            # combined = ft.reduce(instead_of_lambda, results, {})
     
            # lookup_table.clear()
            # lookup_table.update(combined)
            # combined = None

            lookup_table = ft.reduce(instead_of_lambda, results, {})

    with open(f'{lookup_table_name}{layer_level:03d}.dct', 'wb') as fout:
        pic.dump(lookup_table,fout)
        fout.close()
    
    predicted_workers_count = possible_workers_count() 

    if not keep_lookup_table:
        lookup_table = None    
    
    gc.collect()

    print(f'layer {layer_level+1:03d} of size {capacity:>10} complete by {actual_workers_count} worker(s) at {time.time() - start_time:8.2f} sec')
    return predicted_workers_count, lookup_table


def DP_solver_layered(G, clusters, tree, lookup_table_name, need_2_make_layers, workers_count):
    if need_2_make_layers:
        make_layers(clusters,tree, lookup_table_name, workers_count)   

    print('================================')

    num_of_layers = len(clusters) - 1

    predicted_workers_count = possible_workers_count()

    for layer_level in range(num_of_layers):
        keep_lookup_table = (layer_level >= num_of_layers - 1)
        predicted_workers_count, lookup_table = compute_Bellman_layer(G, clusters,  layer_level, tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count)
        
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