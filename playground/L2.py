#
# Построить граф L2
# (расстояния между кластерами через промежуточный кластер)
#
from re import S
import numpy as np
import networkx as nx

from klasses import Task


class L2data:
  """Предрассчитанные данные для быстрого вычисления оценки L2
  """

  def __init__(self, task: Task, start_cluster=1):
    self.task = task

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
        for (p, q), _ in np.ndenumerate(D):
          if dists.has_edge(clusters[A][p], clusters[B][q]):
            D[p, q] = dists.edges[clusters[A][p], clusters[B][q]]['weight'] / 2
        c2p[cid[A], slice(*crgs[B])] = D.min(axis=0)
        p2c[slice(*crgs[A]), cid[B]] = D.min(axis=1)

    self.c2p = c2p
    self.p2c = p2c

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
        for (p, q), _ in np.ndenumerate(D):
          if dists.has_edge(clusters[A][p], clusters[B][q]):
            D[p, q] = dists.edges[clusters[A][p], clusters[B][q]]['weight'] / 2
        c2p[cid[A], slice(*crgs[B])] = D.min(axis=0)
        p2c[slice(*crgs[A]), cid[B]] = D.min(axis=1)

    # Отделим расстояния для стартового кластера
    c2p1 = c2p[:, slice(*crgs[start_cluster])]
    p2c1 = p2c[slice(*crgs[start_cluster]), :]
    di_via_start = np.min(c2p1[..., None] + p2c1[None, ...], axis=1)

    c2p = np.delete(c2p, slice(*crgs[start_cluster]), axis=1)
    p2c = np.delete(p2c, slice(*crgs[start_cluster]), axis=0)
    di = np.min(c2p[..., None] + p2c[None, ...], axis=1)

    # Строим граф L2
    res = nx.DiGraph()
    rez = np.full((len(clusters), len(clusters)), np.inf)
    for A in clusters:
      for B in clusters:
        if A is B:
          continue
        w = np.inf
        if A == start_cluster or B == start_cluster or not task.tree_closure.has_edge(B, A):
          w = di[cid[A], cid[B]]
        if not task.tree_closure.has_edge(A, B):
          w = min(w, di_via_start[cid[A], cid[B]])
        res.add_edge(A, B, weight=w)
        rez[cid[A], cid[B]] = w
    task.L2 = res


if __name__ == '__main__':
  import samples

  # z = samples.random(27, 7)
  z = samples.load("e5x_1")
  L2(z)
