#
# Граф точек с порядком обхода групп, заданным префиксом
#
import networkx as nx
import numpy as np

from klasses import Task, STNode

def prefix_graph(node: STNode):
  task = node.task
  dists = task.dists
  clusters = task.clusters
  result = nx.DiGraph()
  for i in range(1, len(node.sigma)):
    result.add_weighted_edges_from(
      (A, Z, dists.edges[A, Z]['weight']) for A, Z in (
        (A, Z)
        for A in clusters[node.sigma[i - 1]]
        for Z in clusters[node.sigma[i]]
        if dists.has_edge(A, Z)))
  if node.is_leaf():
    result.add_weighted_edges_from(
      (A, (Z, '.'), dists.edges[A, Z]['weight']) for A, Z in (
        (A, Z)
        for A in clusters[node.sigma[-1]]
        for Z in clusters[node.sigma[0]]
        if dists.has_edge(A, Z)))
  return result

def distance_matrix(node: STNode):
  import numpy as np

  task = node.task
  clusters = task.clusters
  cityA = clusters[node.sigma[0]]
  cityZ = clusters[node.sigma[-1]]

  result = np.full((len(cityA), len(cityZ)), np.inf)
  graph = prefix_graph(node)

  for (a, z), _ in np.ndenumerate(result):
    try:
      result[a, z] = nx.shortest_path_length(graph, cityA[a], cityZ[z], weight='weight')
    except nx.NetworkXNoPath:
      pass
  node.shortest_path = result.min()
  return result


def upper_bound(node: STNode):
  """Находит точное решение для заданного порядка обхода кластеров
  """
  if not node.is_leaf():
    raise ValueError("Full route required!")
  vertices = node.task.clusters[node.sigma[0]]
  graph = prefix_graph(node)

  def paths():
    for z in vertices:
      try:
        yield nx.shortest_path_length(graph, z, (z, '.'), weight='weight')
      except nx.NetworkXNoPath:
        pass

  return min(paths())


def deltas(node: STNode, cityA, cityB):
  """
  Матрица расстояний между точками двух кластеров
  """
  dists = node.task.dists
  clusters = node.task.clusters
  A = node.task.clusters[cityA]
  B = node.task.clusters[cityB]
  result = np.full((len(A), len(B)), np.inf)
  for (p, q), _ in np.ndenumerate(result):
    if dists.has_edge(A[p], B[q]):
      result[p, q] = dists.edges[A[p], B[q]]['weight']
  return result

def add_delta(distances, deltas):
  """Добавим дистанции к матрице расстояний
  """
  return (distances[..., None] + deltas[None, ...]).min(axis=1)


def dist_m(node: STNode):
  try:
    return node.Pij
  except AttributeError:
    pass

  if len(node.sigma) == 1:
    n0 = len(node.task.clusters[node.sigma[0]])
    result = np.full((n0, n0), np.inf)
    np.fill_diagonal(result, 0)
  else:
    result = add_delta(dist_m(node.parent), deltas(node, node.sigma[-2], node.sigma[-1]))
  node.Pij = result
  node.shortest_path = result.min()
  return result

def new_upper_bound(node: STNode):
  """Находит точное решение для заданного порядка обхода кластеров
  """
  if not node.is_leaf():
    raise ValueError("Full route required!")
  return add_delta(dist_m(node), deltas(node, node.sigma[-1], node.sigma[0])).diagonal().min()

if __name__ == '__main__':
  import samples, children

  task = samples.random(100, 7)
  # z = samples.load("e1x_1")
  root = STNode(task)
  errors = []
  for z in children.subtree(root):
    if len(z.sigma) == 1:
      continue
    print(z.sigma)
    X = distance_matrix(z)
    Y = dist_m(z)
    if not np.all(X==Y):
      print("Oops!")
      errors.append(z.sigma)
    if not z.is_leaf():
      continue
    VX = upper_bound(z)
    VY = new_upper_bound(z)
    if VX != VY:
      print("Oops again!")
      errors.append(z.sigma)
  print()
  print('Errors:', len(errors))
  for err in errors:
    print("\t-", err)

  from timeit import default_timer as timer

  print("Measuring old...", end='\t', flush=True)
  start = timer()
  for z in children.subtree(STNode(task)):
    if len(z.sigma) == 1:
      continue
    distance_matrix(z)
  end = timer()
  print(end - start)

  print("Measuring new...", end='\t', flush=True)
  start = timer()
  for z in children.subtree(STNode(task)):
    if len(z.sigma) == 1:
      continue
    dist_m(z)
  end = timer()
  print(end - start)
