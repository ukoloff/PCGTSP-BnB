
import Instance_generator as ig
import time
import os

from DP_BnB_solver import DP_solver_layered, visited_clusters, get_path_length, MAXINT

UB = 1.00
NEED_2_MAKE_LAYERS = True

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
    res = DP_solver_layered(graph, clusters, tree, lookup_table_name, NEED_2_MAKE_LAYERS, workers_count, UB)

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
    test(200,10,1)
    # test(1000,20,4)
    # test(1750,20,4) at 25893 sec
    # test(1500,20,4) at 4472 sec
    # test(1000,20,4) at 1640 sec
    # test(200,40,4)  at 7059 sec
    

    

        
    
    
  

