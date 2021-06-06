#
# "Чистый" код для подсчёта пути по "внешнему" пути (суффиксу)
# NC + AP
import networkx as nx


def nc0(dists: nx.DiGraph, clusters: dict, tree: nx.DiGraph, start_cluster=1):
    """Построить граф кластеров для данной задачи
    (с частичным учётом ограничений предшествования)
    """
    # tree = nx.transitive_reduction(order)
    tree_closure = nx.transitive_closure_dag(tree)

    res = nx.DiGraph()
    for A in tree:
        for B in tree:
            if A is B:
                continue
            if B == start_cluster:
                if tree.out_degree(A) != 0:
                    continue
            else:
                if tree_closure.has_edge(B, A):
                    continue
                if tree_closure.has_edge(A, B) and not tree.has_edge(A, B):
                    continue
            w = min(w for w in
                    (dists.edges[cityA, cityB]['weight']
                     for cityA in clusters[A]
                     for cityB in clusters[B]
                     if dists.has_edge(cityA, cityB))
                    if w >= 0)
            res.add_edge(A, B, weight=w)
    return res


def subgraph(nc0: nx.DiGraph, sigma, last_cluster, start_cluster=1):
    """Выкинуть из графа кластеров все кластеры префикса
    (кроме первого и последнего)
    """
    result = nc0.copy()
    if len(sigma) <= 1:
        return result
    result.remove_nodes_from(
        n for n in sigma if not n in (start_cluster, last_cluster))
    result.remove_edges_from([(start_cluster, n)
                             for n in list(result.successors(start_cluster))])
    result.remove_edges_from([(n, last_cluster)
                             for n in list(result.predecessors(last_cluster))])
    # if len(sigma) < len(nc0):
    #     # Not a leaf node
    #     try:
    #         result.remove_edge(last_cluster, start_cluster)
    #     except nx.NetworkXError:
    #         pass
    result.add_edge(start_cluster, last_cluster, weight=0)
    return result


def AP(graph: nx.DiGraph):
    """Рассчитать LB-оценку методом AP / Matching
    """
    bi = nx.Graph()
    for u, v, w in graph.edges.data('weight'):
        bi.add_edge((1, u), (2, v), weight=w)
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bi,
                                                                             top_nodes=((1, n) for n in graph))
    return sum(graph[u][v]['weight'] for (p, u), (q, v) in matching.items() if p == 1 and q == 2)


def lower_bound(nc0: nx.DiGraph, sigma, last_cluster, start_cluster=1):
    """Рассчитать LB для "внешнего" графа
    """
    return AP(subgraph(nc0, sigma, last_cluster, start_cluster))


if __name__ == '__main__':
    import samples
    from salmanize_test import ex5s
    import time

    start_time = time.time()

    # task = samples.random(27, 7)
    task = samples.load("e5x_1")
    cg = nc0(task.dists, task.clusters, task.tree)
    for sigma, ap in ex5s.items():
        LB = lower_bound(cg, sorted(sigma), sigma[-1], sigma[0])
        if LB != ap:
            print("Error for", sigma)
    print("Tested:", len(ex5s), "prefixes")
    print(f"All done in {time.time() - start_time:.02f} sec")
