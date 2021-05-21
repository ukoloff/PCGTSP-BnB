#
# Сгенерировать NC для заданного префикса
#
# Уже nc0 уже должно быть посчитано (task.initialNC)
#
import networkx as nx

from klasses import Task, STNode

def nc(node: STNode):
  result = node.task.initialNC.copy()
  for i in range(1, len(node.sigma) - 1):
    result.remove_node(node.sigma[i])
  if not node.is_leaf():
    try:
      result.remove_edge(node.sigma[-1], node.sigma[0])
    except nx.NetworkXError:
      pass
  result.add_edge(node.sigma[0], node.sigma[-1], weight=node.shortest_path)
  return result

def MSAP(graph):
  """Рассчитать LB-оценку методом MSAP
  """
  msap = nx.minimum_spanning_arborescence(graph)
  return sum(graph[u][v]['weight']
    for u, v in msap.edges) + min(w
    for u, v, w in graph.edges.data('weight'))


def lower_bounds(node: STNode):
  prefix.distance_matrix(node)
  node.bounds = {}
  g = nc(node)
  node.bounds['MSAP'] = MSAP(g)

if __name__ == '__main__':
  import samples, nc0, children, prefix

  z = samples.random(27, 7)
  nc0.nc0(z)
  # z = samples.load("e1x_1")
  root = STNode(z)
  for z in children.subtree(root):
    lower_bounds(z)
    print(z.sigma, z.bounds)
