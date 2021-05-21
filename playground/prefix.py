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

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  # z = samples.load("e1x_1")
  node = STNode(z, (1, 2, 3))
  w = prefix_graph(node)
  print(*w.edges.data())

