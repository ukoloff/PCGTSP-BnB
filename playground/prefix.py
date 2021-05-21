#
# Граф точек с порядком обхода групп, заданным префиксом
#
import networkx as nx

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

if __name__ == '__main__':
  import samples, children

  z = samples.random(27, 7)
  # z = samples.load("e1x_1")
  root = STNode(z)
  for z in children.subtree(root):
    print(z.sigma, distance_matrix(z))
