#
# Сгенерировать NC для заданного префикса
#
# Уже nc0 уже должно быть посчитано (task.initialNC)
#
import networkx as nx

from klasses import Task, STNode
import prefix, nb

# historySuffix[] @ page 9
history = {}

def nc(node: STNode):
  result = node.task.initialNC.copy()
  if len(node.sigma) <= 1:
    return result
  for i in range(1, len(node.sigma) - 1):
    result.remove_node(node.sigma[i])
  result.remove_edges_from((node.sigma[0], n) for n in list(result.successors(node.sigma[0])))
  result.remove_edges_from((n, node.sigma[-1]) for n in list(result.predecessors(node.sigma[-1])))
  if not node.is_leaf():
    try:
      result.remove_edge(node.sigma[-1], node.sigma[0])
    except nx.NetworkXError:
      pass
  result.add_edge(node.sigma[0], node.sigma[-1], weight=0)
  return result

def MSAP(graph):
  """Рассчитать LB-оценку методом MSAP
  """
  msap = nx.minimum_spanning_arborescence(graph)
  return sum(graph[u][v]['weight'] for u, v in msap.edges)

def AP(graph):
  """Рассчитать LB-оценку методом AP / Matching
  """
  bi = nx.Graph()
  for u, v, w in graph.edges.data('weight'):
      bi.add_edge((1, u), (2, v), weight=w)
  matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(bi,
    top_nodes=((1, n) for n in graph))
  return sum(graph[u][v]['weight'] for (p, u), (q, v) in matching.items() if p == 1 and q == 2)


def lower_bounds(node: STNode):
  prefix.distance_matrix(node)
  node.bounds = {}
  S = node.S()
  if S in history:
    node.bounds['LB'] = history[S] + node.shortest_path
    return
  g = nc(node)
  # node.bounds['MSAP'] = MSAP(g)
  node.bounds['AP'] = AP(g)

  # Noon-Bean
  # g = nb.noon_bean(node)
  # node.bounds['NB-MSAP'] = MSAP(g)
  # node.bounds['NB-AP'] = AP(g)

  history[S] = max(node.bounds.values())
  node.bounds['LB'] = history[S] + node.shortest_path


def upper_bound(node: STNode):
  node.bounds = {'UB': prefix.upper_bound(node)}

def bounds(node: STNode):
  if node.is_leaf():
    upper_bound(node)
  else:
    lower_bounds(node)


if __name__ == '__main__':
  import samples, nc0, children

  z = samples.random(27, 7)
  nc0.nc0(z)
  # z = samples.load("e1x_1")
  root = STNode(z)
  for z in children.subtree(root):
    bounds(z)
    print(z.sigma, z.bounds)
