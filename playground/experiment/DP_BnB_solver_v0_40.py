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
from salmanize import lower_bound, lower_bound_harder, precalculate

MAXINT = 100000000000000000000000000000
MAXTASKSPERWORKER = 1000
MEMORY_LIMIT = MAXINT  # MEMORY LIMIT (in Bytes), for cluster calculations

LOW_PC = 0.005
HIGH_PC = 0.99

GAP_TO_STOP = None      # Stop if GAP is reached

def setGap(Gap):
  global GAP_TO_STOP
  if Gap:
    GAP_TO_STOP = Gap


###### Memory usage util #############################################################################
##  ATTENTION: designed for  Linux, for another OS can possible be adapted

MEM_AMPL = 1.5
MEM_RESERVE = 0.7

def possible_workers_count():
    proc = psutil.Process(os.getpid())
    used_this_process = proc.memory_info()[0] * MEM_AMPL
    available = min(MEMORY_LIMIT - used_this_process, psutil.virtual_memory()[1]) * MEM_RESERVE
    return max(1, int(available / used_this_process))



####### We assume that our precedence constraints are defined by a spanning tree rooted at V_1 #######

class State:
    def __init__(self,sigma,j,tilde_u,v, cost = MAXINT, P2_cost = MAXINT, LB = -MAXINT):
        self.sigma = sigma       # index set of visited clusters except the first one
        self.j = j               # index of the last cluster
        self.tilde_u = tilde_u   # last node in the last cluster
        self.v = v               # first node
        self.predecessor_witness = None    # predecessor state witness
        self.cost = cost
        self.P2_cost = P2_cost
        self.LB = LB

    def witness(self):
        return (tuple(self.sigma), self.j, self.tilde_u, self.v)
        # return f'{self.sigma}, {self.j}, {self.tilde_u}, {self.v}'

KEY_SIGMA = 0
KEY_V = 3

KEY_BOUND_SIGMA = 0
KEY_BOUND_VJ = 1
KEY_BOUND_VAL = 2

KEY_RESULTS_CAPACITY = 0
KEY_RESULTS_STATE = 1
KEY_RESULTS_CUTOFF = 2
KEY_RESULTS_LB = 3


def make_unique_list(lst):
    tuples=map(lambda item: tuple(sorted(item)),lst)
    cnt = Counter(tuples)
    unique_tuples = list(cnt.keys())
    return list(map(list,unique_tuples))


def compute_Bellman_cell(G, clusters, transitive_closure, lookup_table, filtered_prev_state_keys, state):
    best_cost = MAXINT
    prev_state = None


    for key in filtered_prev_state_keys:
        suggested_prev_state = mp_lookup_table[key]
        cost = suggested_prev_state.cost + G[suggested_prev_state.tilde_u][state.tilde_u]['weight']
        if cost < best_cost:
            best_cost = cost
            prev_state = suggested_prev_state


    if best_cost < MAXINT:
        state.cost = best_cost
        state.predecessor_witness = prev_state.witness()
        # lookup_table[state.witness()] = state

    return best_cost

################ multiprocessing staff #########
###
def worker_init(G, clusters, tree, current_layer, UB, lookup_table_filename = ''):
    global mp_G
    global mp_clusters
    global mp_tree
    global mp_lookup_table
    global mp_transitive_closure
    global mp_precalcutated
    global mp_UB
    global mp_succs
    global mp_current_layer
    global mp_start_cluster_id


    mp_G, mp_clusters, mp_tree = G, clusters, tree
    mp_transitive_closure = nx.transitive_closure(tree)
    mp_UB = UB
    mp_current_layer = current_layer

    mp_succs={}

    c_keys=list(mp_clusters.keys())
    mp_start_cluster_id = c_keys[0]

    # mp_nc0 = nc0(mp_G, mp_clusters, mp_tree, start_cluster_id)
    mp_precalcutated = precalculate(mp_G, mp_clusters, mp_tree, mp_start_cluster_id)
    # print(f'NC0=\n{mp_nc0.edges(data="weight")}')

    mp_lookup_table = {}

    if lookup_table_filename:
        with open(lookup_table_filename, 'rb') as fin:
            mp_lookup_table = pic.load(fin)


######
######
##
##   Find the best among of fast-computable bounds for the P2 Salman's problem
##   INPUT:  tuple(sigma, list of possible finishing clusters (Vjs))
##   OUTPUT: [(sigma, Vj, P2_cost) ... ]
##
######
######

def parallel_fast_P2_bounds(sigma_Vjs):
    sigma, Vjs = sigma_Vjs
    l_sigma=list(sigma)
    result = []

    for ind_V_j in Vjs:
        P2_cost = 0
        ###
        try:
            P2_cost = lower_bound(mp_precalcutated, [mp_start_cluster_id] + l_sigma, ind_V_j, mp_start_cluster_id)
        except:
            print(f'LB fast bounding fault, sigma={sigma}, start={mp_start_cluster_id}, dest={ind_V_j}')
        ###
        result.append((sigma, ind_V_j, P2_cost))
    return result


######
######
##
##   Try to solve the P2 Salman's problem exactly
##   INPUT:  tuple(sigma, Vj, raw_P2_cost)
##   OUTPUT: tuple(sigma, Vj, revised_P2_cost)
##
######
######

def parallel_improve_P2_bound(sigma_Vj_rawP2bound):
    sigma, ind_V_j, raw_P2_bound = sigma_Vj_rawP2bound
    c_keys=list(mp_clusters.keys())
    start_cluster_id = c_keys[0]
    l_sigma = list(sigma)


    gurobi_bound = raw_P2_bound
    try:
        gurobi_bound = lower_bound_harder(mp_precalcutated, [mp_start_cluster_id] + l_sigma, ind_V_j, mp_start_cluster_id)
    except:
        print(f'LB Gurobi bounding fault, sigma={sigma}, start={mp_start_cluster_id}, dest={ind_V_j}')

    return sigma, ind_V_j, max(gurobi_bound, raw_P2_bound)


######
######
##
##   Construct a part of the novel lookup table layer
##   INPUT:  tuple(sigma, Vj, P2_cost)
##   OUTPUT: (capacity, result, cutoff, localLB)
##
##      capacity    number of produced states
##      cutoff      number of pruned states / branches
##      results     list(state1, state2, ...)
##
##
##      localLB     min among P2_cost + <length of the partial path defined by the state>
##
######
######

def parallel_layer_2_list(sigma_Vj_P2cost):
    global mp_lookup_table

    sigma, ind_V_j, P2_cost = sigma_Vj_P2cost

    result =[]
    capacity = 0
    cutoff = 0
    localLB = MAXINT

    sigma = list(sigma)


    truncated_sigma = tuple([ind for ind in sigma if ind != ind_V_j])
    ts_keys = [key for key in mp_lookup_table if key[KEY_SIGMA] == truncated_sigma]

    for v in mp_clusters[mp_start_cluster_id]:
        filtered_prev_state_keys = [key for key in ts_keys if key[KEY_V] == v]
        for tilde_u in mp_clusters[ind_V_j]:
            state = State(sigma, ind_V_j, tilde_u, v)
            state.cost = compute_Bellman_cell(mp_G, mp_clusters, mp_transitive_closure, mp_lookup_table, filtered_prev_state_keys, state)
            if state.cost < MAXINT:
                state.LB = P2_cost + state.cost
                if state.LB <= mp_UB:
                    state.P2_cost = P2_cost
                    result.append(state)
                    capacity += 1
                    localLB = min(localLB, state.LB)
                else:
                    cutoff += 1
    return capacity, result, cutoff, localLB

###
##################################################




def do_prepare_layer(sigma, tree, V1, V1_succ, current_layer):
    l_sigma = list(sigma)
    suppl = [(tuple(sorted(l_sigma + [succ])),succ) for succ in V1_succ if succ not in l_sigma]
    for new_sigma, succ in suppl:
        if not new_sigma in current_layer:
            current_layer[new_sigma] = [succ]
        current_layer[new_sigma].append(succ)


    for ind in l_sigma:
        suppl = [(tuple(sorted(l_sigma + [succ])),succ) for succ in tree.successors(ind)
        if succ not in l_sigma and nx.ancestors(tree,succ).issubset(l_sigma+[V1])]

        for new_sigma, succ in suppl:
            if not new_sigma in current_layer:
                current_layer[new_sigma] = [succ]
            current_layer[new_sigma].append(succ)


def prepare_layer(clusters,tree, previous_layer, layer_level):
    clust_keys=list(clusters.keys())
    V1 = clust_keys[0]
    V1_succ = list(tree.successors(V1))
    start_time = time.time()

    current_layer = {}

    if not previous_layer:     # baseline case
        current_layer = {(succ,): None for succ in V1_succ}
        return current_layer

    for sigma in previous_layer:
        do_prepare_layer(sigma, tree, V1, V1_succ, current_layer)

    current_layer = {key: list(set(val)) for key,val in current_layer.items()}
    return current_layer



##########
##########
##
##  Non-optimal version of the dict{(sigma, Vj, P2) -> list(state1, state2, ...) for P2 bound revision
##  Relying on shallow copying of the states
##
##########
##########

def prepare_Uranus(states_list, low_percent=LOW_PC, high_percent=HIGH_PC):
    L = len(states_list)
    low_count = int(L * low_percent)
    high_count = int(L* high_percent)

    to_revise_dict = {}
    to_revise =set()
    for state in states_list[:low_count] + states_list[high_count:]:
        long_key = (tuple(state.sigma), state.j, state.P2_cost)
        short_key = (tuple(state.sigma), state.j)

        if not short_key in to_revise_dict:
            to_revise_dict[short_key] = [state]
        else:
            to_revise_dict[short_key].append(state)

        to_revise.add(long_key)

    return to_revise_dict, list(to_revise)



def compute_Bellman_layer(G, clusters,  layer_level, previous_layer, tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count, UB, LB):
    c_keys=list(clusters.keys())
    start_time = time.time()

    layer = prepare_layer(clusters, tree, previous_layer, layer_level)
    print(f'layer {layer_level+1:03d} is prepared')

    lookup_table = {}
    start_cluster_id = c_keys[0]

    capacity = 0
    cutoff = 0
    layerLB = MAXINT

    if layer_level < 1: # baseline case - we use the short run
        actual_workers_count = 1
        for sigma in layer:
            sigma = list(sigma)
            for ind_V_j in sigma:
                precalculated = precalculate(G, clusters, tree, start_cluster_id)
                P2_cost = 0
                try:
                    P2_cost = lower_bound(precalculated, [start_cluster_id] + sigma, ind_V_j, start_cluster_id)
                except:
                    print(f'LB fast bounding fault, sigma={sigma}, start={start_cluster_id}, dest={ind_V_j}')

                revised_bound = P2_cost
                try:
                    revised_bound = lower_bound_harder(precalculated, [start_cluster_id] + sigma, ind_V_j, start_cluster_id)
                except:
                    print(f'LB Gurobi bounding fault, sigma={sigma}, start={start_cluster_id}, dest={ind_V_j}')


                for v in clusters[start_cluster_id]:
                    for tilde_u in clusters[ind_V_j]:
                        if G.has_edge(v,tilde_u):             # checking if graph has the edge (v,tilde_u)
                            capacity += 1
                            state = State(sigma, ind_V_j, tilde_u, v)
                            state.cost = G[v][tilde_u]["weight"]
                            state.LB = state.cost + revised_bound
                            layerLB = min(layerLB, state.LB)

                            witness = state.witness()
                            lookup_table[witness] = state


    else:

        actual_workers_count = min(workers_count, predicted_workers_count)

        ### compute raw lower bounds for P2
        with mp.Pool(actual_workers_count, worker_init,(G, clusters, tree, layer, UB), maxtasksperchild=MAXTASKSPERWORKER) as pool:


            chunks_fast_P2_bounds = pool.map(parallel_fast_P2_bounds, layer.items())
            pool.close()
            pool.join()

            # collect raw results from the workers
            fast_P2_bounds = [t for chunk in chunks_fast_P2_bounds for t in chunk]

            print(f'raw P2 bounds calculated')

        ### compute raw states
        with mp.Pool(actual_workers_count, worker_init,(G, clusters, tree, layer, UB, f'{lookup_table_name}{layer_level-1:03d}.dct'), maxtasksperchild=MAXTASKSPERWORKER) as pool:
            results = pool.map(parallel_layer_2_list, fast_P2_bounds)
            pool.close()
            pool.join()

            pool = None
            layer = None

            capacity = sum(map(lambda item: item[KEY_RESULTS_CAPACITY],results))
            cutoff = sum(map(lambda item: item[KEY_RESULTS_CUTOFF],results))
            # layerLB = min(map(lambda item: item[KEY_RESULTS_LB],results))

            ### combine and sort states by local LBs
            states_list = [s for chunk in results for s in chunk[KEY_RESULTS_STATE]]
            states_list.sort(key = lambda state: state.LB)

            ### take a given part of states for the revision
            to_revise_dict, to_revise = prepare_Uranus(states_list)

            print(f'raw state costs computed, {len(to_revise)} P2 bounds sent for revision')

        if to_revise:
            with mp.Pool(actual_workers_count, worker_init,(G, clusters, tree, layer, UB), maxtasksperchild=MAXTASKSPERWORKER) as pool:
                revised = pool.map(parallel_improve_P2_bound, to_revise)
                pool.close()
                pool.join()
                # collect revised P2 bounds from the workers

                print(f'revised P2 bounds obtained')

                # recompute the states
                # revised_layerLB = MAXINT

                sum_diff = 0
                count = 0

                for item in revised:
                    for state in to_revise_dict[(item[KEY_BOUND_SIGMA], item[KEY_BOUND_VJ])]:
                        old_localLB = state.LB
                        state.P2_cost = item[KEY_BOUND_VAL]
                        state.LB =  state.cost + state.P2_cost

                        assert state.LB >= old_localLB, f'Error: raw LB {old_localLB} turns to be better than revised {state.LB}'
                        sum_diff += state.LB - old_localLB
                        count += 1


                        # if revised_layerLB > state.LB:
                        #     revised_layerLB = state.LB

                print(f'Average local LB growth is {sum_diff / count:.3f}')

        ### construct current layer
        lookup_table = {state.witness(): state for state in states_list}

        # layerLB = max(layerLB, revised_layerLB)

        layerLB = min([state.LB for state in states_list])


    with open(f'{lookup_table_name}{layer_level:03d}.dct', 'wb') as fout:
        pic.dump(lookup_table,fout)
        fout.close()

    ####### to filter out the next layer ###################
    sigmas_from_lookup_table = list(set([key[KEY_SIGMA] for key in lookup_table]))
    ########################################################

    predicted_workers_count = possible_workers_count()

    if not keep_lookup_table:
        lookup_table = None

    gc.collect()

    LB = max(LB, layerLB)
    Gap = (UB - LB)/LB


    print(f'\t{cutoff} ({cutoff / (cutoff + capacity):.1%}) of branches were cut off')
    print(f'layer {layer_level+1:03d} of size {capacity:>10} ({len(sigmas_from_lookup_table)}) is completed by {actual_workers_count} worker(s) at {time.time() - start_time:8.2f} sec.')

    print(f'Best UB is {UB}, Best LB is {LB}, Gap is {Gap:.2%}\n=============================')
    return predicted_workers_count, lookup_table, sigmas_from_lookup_table, LB


def DP_solver_layered(G, clusters, tree, lookup_table_name, need_2_keep_layers, workers_count, UB = MAXINT):
    start_time = time.time()

    num_of_layers = len(clusters) - 1

    predicted_workers_count = possible_workers_count()
    previous_layer = []

    c_keys=list(clusters.keys())
    start_cluster_id = c_keys[0]

    precalculated = precalculate(G, clusters, tree, start_cluster_id)
    LB = lower_bound(precalculated, [start_cluster_id], start_cluster_id, start_cluster_id)

    print(f'Start LB is {LB}')
    print(f'Starting layers at: {time.time() - start_time:8.2f}')

    def cleanUpTables():
      if not need_2_keep_layers:
          filelist = glob.glob(f'{lookup_table_name}*.dct', recursive = False)
          for ff in filelist:
              try:
                  os.remove(ff)
              except OSError as mag:
                  print(msg)

    if GAP_TO_STOP and GAP_TO_STOP >= (UB - LB) / LB * 100:
      print(f"GAP of {GAP_TO_STOP}% is met!")
      cleanUpTables()
      return (UB, [], [])


    for layer_level in range(num_of_layers):
        keep_lookup_table = (layer_level >= num_of_layers - 1)
        predicted_workers_count, lookup_table, previous_layer, LB = compute_Bellman_layer(G, clusters,  layer_level, previous_layer,
            tree, lookup_table_name, keep_lookup_table, workers_count, predicted_workers_count, UB, LB)
        if GAP_TO_STOP and GAP_TO_STOP >= (UB - LB) / LB * 100:
          print(f"GAP of {GAP_TO_STOP}% is met!")
          cleanUpTables()
          return (UB, [], [])

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

    while current_state.predecessor_witness != None:
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

    cleanUpTables()

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
