
import networkx as nx
import Instance_generator as ig
import time
import multiprocessing as mp
import pickle as pic
import gc
import psutil
import os
import functools as ft

from collections import Counter


MAXINT = 100000000000000000000000000000
MAXTASKSPERWORKER = 1000

###### Memory usage util #############################################################################
##  ATTENTION: designed for the Linux, for another OS can be possible adapted

MEM_AMPL = 1.5
MEM_RESERVE = 0.8

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

    def retrieve_from_cached(self, src):
        self.predec = src.predec
        self.cost = src.cost
    

def make_unique_list(lst):
    tuples=map(tuple, lst)
    cnt = Counter(tuples)
    unique_tuples = list(cnt.keys())
    return list(map(list,unique_tuples))

def add_layer(clusters,tree,previous_layer):
    current_layer=[]
    clust_keys=list(clusters.keys())
    V1_succ = tree.successors(clust_keys[0])

    if len(previous_layer) == 0:     # baseline case 
        current_layer = [[succ] for succ in V1_succ]
        return current_layer

    for sigma in previous_layer:    
        l_sigma = list(sigma) 
        current_layer += [(l_sigma + [succ]) for succ in V1_succ if succ not in sigma]
    
        for ind in sigma:
            suppl = [l_sigma + [succ] for succ in tree.successors(ind) if succ not in sigma]
            suppl = list(map(sorted, suppl))
            current_layer += suppl

    return make_unique_list(current_layer)

def make_layers(clusters,tree):
    num_of_layers = len(clusters) - 1
    layers=[]
    previous_layer = []
    for ind in range(num_of_layers):
        current_layer = add_layer(clusters, tree, previous_layer)
        layers.append(current_layer)
        previous_layer = current_layer
    return layers

def can_be_the_last_cluster(sigma,c_ind, tree):
    succ = tree.successors(c_ind)
    return all(not s in sigma for s in succ)

def compute_Bellman_cell(G, clusters, tree, lookup_table, state):
    witness = state.witness()
    if witness in lookup_table:
        cached_state = lookup_table[witness]
        state.retrieve_from_cached(cached_state)
        return state.cost

    best_cost = MAXINT
    prev_state = None

    truncated_sigma = [ind for ind in state.sigma if ind != state.j]

    for ind_prev_cluster in [ind for ind in truncated_sigma if can_be_the_last_cluster(truncated_sigma,ind,tree)]:
        for tilde_u in clusters[ind_prev_cluster]:
            suggested_prev_state = State(truncated_sigma,ind_prev_cluster,tilde_u,state.v)
            witness = suggested_prev_state.witness()
            if witness in lookup_table:
                pretender_state = lookup_table[witness]
                cost = pretender_state.cost + G[pretender_state.tilde_u][state.tilde_u]['weight']
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

    mp_G, mp_clusters, mp_tree = G, clusters, tree

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

    for ind_V_j in [ind for ind in sigma if can_be_the_last_cluster(sigma, ind, mp_tree)]:
        for v in mp_clusters[c_keys[0]]:
            for tilde_u in mp_clusters[ind_V_j]:
                state = State(sigma, ind_V_j, tilde_u, v)
                state.cost = compute_Bellman_cell(mp_G, mp_clusters, mp_tree, mp_lookup_table, state)
                if state.cost < MAXINT:
                    result[state.witness()] = state
                    capacity += 1
    return (capacity, result)


def compute_Bellman_layer(G, clusters, layers, layer_level, tree, lookup_table_name, lookup_table, workers_count):
    layer = layers[layer_level]
    c_keys=list(clusters.keys())
    start_time = time.time()


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
                            lookup_table[state.witness()] = state
                
    else:

        actual_workers_count = min(workers_count, possible_workers_count())
        # print(f'{actual_workers_count} workers')
        # print('create pool')
        pool = mp.Pool(actual_workers_count, worker_init, (G, clusters, tree, f'{lookup_table_name}{layer_level-1:03d}.dct'), maxtasksperchild=MAXTASKSPERWORKER)
        # print('mapping')
        results = pool.map(parallel, layer)
        pool.close()
        pool.join()
        # print('complete')

        capacity = sum(map(lambda item: item[0],results))
        # print(f'capacity: {capacity}')
        # combined = ft.reduce(lambda acc, res: {**acc, **(res[1])}, results, {})

        def instead_of_lambda(acc, res):
            acc.update(res[1])
            return acc

        combined = ft.reduce(instead_of_lambda, results, {})
        # print('reduction complete')

        lookup_table.clear()
        lookup_table.update(combined)

    with open(f'{lookup_table_name}{layer_level:03d}.dct', 'wb') as fout:
        pic.dump(lookup_table,fout)
        fout.close()
        # print('table dumped')

    gc.collect()

    print(f'layer {layer_level+1:03d} of size {capacity:>8} complete by {actual_workers_count} workers at {time.time() - start_time:8.2f} sec')


def DP_solver_layered(G, clusters, tree, lookup_table_name, workers_count):
    layers = make_layers(clusters,tree)
    
    lookup_table = {}
    for layer_level in range(len(layers)):
        compute_Bellman_layer(G, clusters, layers, layer_level, tree, lookup_table_name, lookup_table, workers_count)

    OPT = MAXINT
    best_state = None

    leaves = [ind_V for ind_V in clusters if tree.out_degree(ind_V) == 0]

    clust_keys = list(clusters.keys())
    sigma = sorted(clust_keys[1:])
    ind_V_1 = clust_keys[0]

    for v in clusters[ind_V_1]:
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


      

def test(n,m, workers_count):
    graph = ig.graph_generator(n)
    clusters = ig.clustering(n,m)
    tree = ig.tree_gen(m)
    order = ig.complete_order(tree)
    tour = ig.create_opt_tour(clusters, order)
    graph = ig.update_graph(graph,tour)
    print(f'Number of nodes: {n}, Number of clusters: {m}')
    print(f'Graph: {graph.edges(data = "weight")}')
    print(f'Clusters: {clusters}')
    print(f'Partial order: {tree.edges()}')
    
    start_time = time.time()
    
    lookup_table_name = f'problem_{n}_{m}_'
    res = DP_solver_layered(graph, clusters, tree, lookup_table_name, workers_count)

    print(f'Expected order: {order}')
    print(f'Expected optimal tour: {tour + [tour[0]]}')
    print(f'RESULT: ')
    if res:
        print(f'OPT = {res[0]}')
        print(f'Optimal Tour: {res[2]}')
        print(f'Visited clusters: {res[1]}')
    else:
        print('Instance is infeasible')
    print(f'Elapsed time: {time.time()-start_time:.2f} sec')


if __name__ == '__main__':
    test(1000,20,4)
    # test(1750,20,4) at 25893 sec
    # test(1500,20,4) at 4472 sec
    # test(1000,20,4) at 1640 sec
    # test(200,40,4)  at 7059 sec
    

    

        
    
    
  

