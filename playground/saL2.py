#
# "Чистый" код для подсчёта пути по "внешнему" пути (суффиксу)
# в режиме L2
#
import numpy as np
import networkx as nx


class L2data:
    """Предрассчитанные данные для быстрого вычисления оценки L2
    """

    def __init__(self, dists: nx.DiGraph, clusters: dict, tree: nx.DiGraph, tree_closure: nx.DiGraph, start_cluster=1):
        """
        dists - расстояния между точками
        clusters - распределение точек по кластерам
        tree - дерево порядка (редуцированное!)
        start_cluster - начальный кластер
        """

        # tree_closure = nx.transitive_closure_dag(tree)
        self.tree_closure = tree_closure
        self.clusters = clusters

        # Индексы
        cid = {c: i for i, c in enumerate(clusters)}
        lengths = [len(c) for c in clusters.values()]
        rgs = [(end - len, end)
               for len, end in zip(lengths, np.cumsum(lengths))]
        crgs = {c: rgs[cid[c]] for c in clusters}

        self.cid = cid
        self.crgs = crgs

        # Расстояния от точек до кластеров и от кластеров до точек
        c2p = np.full((len(clusters), len(dists)), np.inf)
        p2c = np.full((len(dists), len(clusters)), np.inf)

        for A in clusters:
            for B in clusters:
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
                D = np.full((len(clusters[A]), len(clusters[B])), np.inf)
                for (p, q), _ in np.ndenumerate(D):
                    if dists.has_edge(clusters[A][p], clusters[B][q]):
                        D[p, q] = dists.edges[clusters[A][p],
                                              clusters[B][q]]['weight'] / 2
                c2p[cid[A], slice(*crgs[B])] = D.min(axis=0)
                p2c[slice(*crgs[A]), cid[B]] = D.min(axis=1)

        self.c2p = c2p
        self.p2c = p2c

    def suffix_graph(self, sigma, last_cluster, start_cluster=1):
        cid = self.cid
        crgs = self.crgs

        p2c = self.p2c.copy()
        c2p = self.c2p.copy()

        # Уберём пути до кластеров внутри префикса
        omit = set(sigma) - set([last_cluster, start_cluster])
        for g in omit:
            p2c[slice(*crgs[g]), :] = np.inf
            c2p[:, slice(*crgs[g])] = np.inf

        # Занулим пути от начала префикса до конца (p2)
        if start_cluster != last_cluster:
            c2p[cid[start_cluster], slice(*crgs[last_cluster])] = 0
            p2c[slice(*crgs[start_cluster]), cid[last_cluster]] = 0

        # Расстояния через 1 кластер
        c2p1 = c2p[:, slice(*crgs[start_cluster])]
        p2c1 = p2c[slice(*crgs[start_cluster]), :]
        di_via_start = np.min(c2p1[..., None] + p2c1[None, ...], axis=1)

        # Расстояния в обход 1 кластера
        c2p = np.delete(c2p, slice(*crgs[start_cluster]), axis=1)
        p2c = np.delete(p2c, slice(*crgs[start_cluster]), axis=0)
        di = np.min(c2p[..., None] + p2c[None, ...], axis=1)

        # Строим граф L2
        tree_closure = self.tree_closure
        clusters = self.clusters
        res = nx.DiGraph()
        # rez = np.full((len(clusters), len(clusters)), np.inf)
        for A in clusters:
            if A in omit:
                continue
            for B in clusters:
                if A is B:
                    continue
                if B in omit:
                    continue
                w = np.inf
                if A == start_cluster or B == start_cluster or not tree_closure.has_edge(B, A):
                    w = di[cid[A], cid[B]]
                if not tree_closure.has_edge(A, B):
                    w = min(w, di_via_start[cid[A], cid[B]])
                if np.isinf(w):
                    continue
                res.add_edge(A, B, weight=w)
                # rez[cid[A], cid[B]] = w
        return res


if __name__ == '__main__':
    import samples

    task = samples.load("e5x_1")
    z = L2data(task.dists, task.clusters, task.tree)
    z.suffix_graph((1, 2, 3), 3, 1)
