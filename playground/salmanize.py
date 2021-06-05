#
# "Чистый" код для подсчёта пути по "внешнему" пути (суффиксу)
# NC + APa
import networkx as nx


def nc0(dists: nx.DiGraph, clusters: dict, order: nx.DiGraph):
    """Построить граф кластеров для данной задачи
    (с частичным учётом ограничений предшествования)
    """
    tree = nx.transitive_reduction(order)
    tree_closure = nx.transitive_closure_dag(tree)

    res = nx.DiGraph()
    for A in tree:
        for B in tree:
            if A is B:
                continue
            if B == 1:
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


if __name__ == '__main__':
    import samples

    # task = samples.random(27, 7)
    task = samples.load("e5x_1")
    cg = nc0(task.dists, task.clusters, task.tree)
    print(cg)
