
import networkx as nx
import Instance_generator as ig
import time
import os
import sys


from fromPCGLNS import getInstance
from DP_BnB_solver_v0_21 import DP_solver_layered, visited_clusters, get_path_length, MAXINT, MEMORY_LIMIT

      

def test(filename, need_2_keep_layers, workers_count, UB):
    
    graph, clusters, tree = getInstance(filename)

    n = graph.number_of_nodes()
    m = len(clusters)

    print(f'Number of nodes: {n}, Number of clusters: {m}')
    print(f'Clusters: {clusters}')
    print(f'Partial order tree: {tree.edges()}')

    start_time = time.time()
    
    lookup_table_name = f'problem_{n}_{m}_'
    res = DP_solver_layered(graph, clusters, tree, lookup_table_name, need_2_keep_layers, workers_count, UB)
    
    print(f'RESULT: ')
    if res:
        OPT, route, tour = res

        print(f'OPT = {OPT}')
        print(f'Optimal Tour: {tour}')
        print(f'Visited clusters: {route}')

        print(f'Tour length (rechecked): {get_path_length(tour, graph)}')


    else:
        print('Instance is infeasible')
    print(f'Elapsed time: {time.time()-start_time:.2f} sec')


def test2(filename):
    graph, clusters, tree = getInstance(filename)
    tour2 = [1, 251, 238, 112, 93, 65, 34, 218, 214, 234, 231, 266, 264, 283, 277, 146, 121, 314, 313, 299, 284, 200, 173, 1]

    print(f'tour: {tour2}')
    print(f'clusters: {visited_clusters(tour2, clusters)}')
    print(f'tour length (rechecked): {get_path_length(tour2, graph)}')

if __name__ == '__main__':
    # test2('../pcglns/e5x_1.pcglns')          # PCGLNS results: Obj val.: 1890 - < 5 sec.
    # test('../pcglns/e5x_1.pcglns',True, 5)   # OPT: 1847 - time 4662 sec

    # test('../pcglns/e3x_2.pcglns',True,6)      # PCGLNS results: Obj val.: 1584 - < 5 sec., OPTL 1578 - time 15824 sec

    # test('../Salman/input/ESC07.pcglns', True, 3)  OPT: 1730, time < 2 sec
    ifname = ''
    workers_count = 1
    UB = MAXINT
    keep_layers = False

    try:
        for arg in sys.argv:
            if arg == '--keep_layers':
                keep_layers = True
            if '=' in arg:
                parts = arg.split('=')
                if parts[0] == '--input' or parts[0] == '-i':
                    ifname = parts[1]
                if parts[0] == '--workers' or parts[0] == '-w':
                    workers_count = int(parts[1])
                if parts[0] == '--upper_bound' or parts[0] == '-UB':
                    UB = float(parts[1])
                if parts[0] == '--memory_limit_gb' or parts[0] == '-m':
                    MEMORY_LIMIT = int(parts[1]) * 1000000000

    except:
        print('SYNTAX: python DP_parallel_layered3_pcglns.py -i=<input path/filename> [-w=<workers_count>] [-UB=<upper_bound>] [-m=<memory_limit (Gb)>]')

    test(ifname, keep_layers, workers_count, UB)

    

    

        
    
    
  

