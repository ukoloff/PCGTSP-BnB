#
# Граф точек с порядком обхода групп, заданным префиксом
#
import networkx as nx
import numpy as np

from klasses import Task, STNode


def deltas(node: STNode, cityA, cityB):
  """
  Матрица расстояний между точками двух кластеров
  """
  dists = node.task.dists
  clusters = node.task.clusters
  A = clusters[cityA]
  B = clusters[cityB]
  result = np.full((len(A), len(B)), np.inf)
  for (p, q), _ in np.ndenumerate(result):
    if dists.has_edge(A[p], B[q]):
      result[p, q] = dists.edges[A[p], B[q]]['weight']
  return result

def add_delta(distances, deltas):
  """Добавим дистанции к матрице расстояний
  """
  return (distances[..., None] + deltas[None, ...]).min(axis=1)


def distance_matrix(node: STNode):
  try:
    return node.Pij
  except AttributeError:
    pass

  if len(node.sigma) == 1:
    # Вырожденная матрица расстояний
    n0 = len(node.task.clusters[node.sigma[0]])
    result = np.full((n0, n0), np.inf)
    np.fill_diagonal(result, 0)
  else:
    result = add_delta(distance_matrix(node.parent), deltas(node, node.sigma[-2], node.sigma[-1]))
  node.Pij = result
  node.shortest_path = result.min()
  return result

def upper_bound(node: STNode):
  """Находит точное решение для заданного порядка обхода кластеров
  """
  if not node.is_leaf():
    raise ValueError("Full route required!")
  return add_delta(distance_matrix(node), deltas(node, node.sigma[-1], node.sigma[0])).diagonal().min()

if __name__ == '__main__':
  import samples, children

  task = samples.random(100, 7)
  # z = samples.load("e1x_1")
  root = STNode(task)
  for z in children.subtree(root):
    distance_matrix(z)
    print(z.sigma, z.shortest_path)
