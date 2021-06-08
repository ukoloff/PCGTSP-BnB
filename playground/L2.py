#
# Построить граф L2
# (расстояния между кластерами через промежуточный кластер)
#
import enum
import numpy as np
import networkx as nx
from numpy.lib import ndenumerate

from klasses import Task


def L2(task: Task, start_cluster=1):
    """Рассчитать мин. расстояния между кластерами
    с шагом через 1 кластер
    (с учётом ограничений предшествования)
    """

    dists = task.dists
    clusters = task.clusters

    # Индексы
    cid = {c: i for i, c in enumerate(clusters)}
    lengths = [len(c) for c in clusters.values()]
    rgs = [(end - len, end) for len, end in zip(lengths, np.cumsum(lengths))]
    crgs = {c: rgs[cid[c]] for c in clusters}

    # Расстояния от точек до кластеров и от кластеров до точек
    c2p = np.full((len(clusters), len(dists)), np.inf)
    p2c = np.full((len(dists), len(clusters)), np.inf)

    for A in clusters:
      for B in clusters:
        if A is B:
          continue
        if B == start_cluster:
          if task.tree.out_degree(A) != 0:
            continue
        else:
          if task.tree_closure.has_edge(B, A):
            continue
          if task.tree_closure.has_edge(A, B) and not task.tree.has_edge(A, B):
            continue
        D = np.full((len(clusters[A]), len(clusters[B])), np.inf)
        for (p, q), _ in ndenumerate(D):
          if dists.has_edge(clusters[A][p], clusters[B][q]):
            D[p, q] = dists.edges[clusters[A][p], clusters[B][q]]['weight']
        c2p[cid[A], slice(*crgs[B])] = D.min(axis=0)
        p2c[slice(*crgs[A]), cid[B]] = D.min(axis=1)
    print(c2p)
    print(p2c)

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  L2(z)
