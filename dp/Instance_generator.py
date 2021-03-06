import networkx as nx
import numpy as np
import random as rand
SEED = 9001
np.random.seed(SEED)
rand.seed(SEED)
LOW_WEIGHT = 10
HIGH_WEIGHT = 20


def graph_generator(n):
    graph = nx.DiGraph()
    weight_matrix = np.random.randint(LOW_WEIGHT, HIGH_WEIGHT, size=(n, n))
    edges = [(i+1, j+1, weight_matrix[i, j])
             for i in range(n) for j in range(n) if i != j]
    graph.add_weighted_edges_from(edges)
    return graph

def random_combination(iterable, r):
    pool = tuple(iterable)
    pool_len = len(pool)
    indices = sorted(rand.sample(range(pool_len), r))
    return tuple(pool[i] for i in indices)

def clustering(n,m):
    tpl = random_combination(range(2,n+1), m-1)
    tpl = (1,*tpl,n+1)
    # print(tpl)
    clusters={}
    for ind in range(1,m+1):
        clusters[ind]=[i for i in range(tpl[ind-1],tpl[ind])]
    return clusters

def tree_gen(m):
    tree = nx.random_tree(m, seed=SEED)
    tree = nx.dfs_tree(tree, source=0)
    ditree = nx.DiGraph()
    ditree.add_edges_from([(u+1, v+1) for (u, v) in tree.edges()])
    # print(ditree.edges())
    return ditree


def complete_order(tree):
    return list(nx.dfs_preorder_nodes(tree, source=1))


def create_opt_tour(clusters, order):
    tour = [clusters[o][0] for o in order]
    return tour


def update_graph(graph, tour):
    for ind in range(len(tour)-1):
        u = tour[ind]
        v = tour[ind+1]
        graph[u][v]["weight"] = 0
    graph[tour[-1]][tour[0]]["weight"] = 0
    return graph

def get_instance(n,m):
    clusters = clustering(n,m)
    tree = tree_gen(m)
    order = complete_order(tree)
    tour = create_opt_tour(clusters, order)
    graph = update_graph(graph_generator(n), tour)

    return graph, clusters, tree

if __name__ == '__main__':
    n=10
    m=3
    clusters = clustering(n,m)
    tree = tree_gen(m)
    order = complete_order(tree)
    tour = create_opt_tour(clusters, order)
    graph = update_graph(graph_generator(n), tour)

    print(f'PCGTSP random instance generator')
    print(f'Number of nodes: {n}')
    print(f'Number of clusters: {m}')
    print(f'Graph: nodes={graph.nodes()}')
    print(f'Graph: edges')
    print(graph.edges(data="weight"))
    print(f'clusters={clusters}')
    print(f'PC DAG: {tree.edges()}')
    print(f'linear order={order}')
    print(f'optimal tour={tour}')
