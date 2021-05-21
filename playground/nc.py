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

if __name__ == '__main__':
  import samples, nc0, children, prefix

  z = samples.random(27, 7)
  nc0.nc0(z)
  # z = samples.load("e1x_1")
  root = STNode(z)
  for z in children.subtree(root):
    prefix.distance_matrix(z)
    NC = nc(z)
    print(z.sigma, *NC.edges.data('weight'))
