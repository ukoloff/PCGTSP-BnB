#
# Cut по префиксу
#
import networkx as nx
import numpy as np

from klasses import Task, STNode
import prefix

# historyPathCosts[] @ page 9
history = {}

def skip(node: STNode):
  """Проверить, что путь по префиксу не слишком длинен
  """
  S = node.S()
  dist = prefix.distance_matrix(node)
  if not S in history:
    history[S] = dist
    return
  if np.all(history[S] <= dist):
    node.skip = True
    return
  history[S] = np.minimum(history[S], dist)

if __name__ == '__main__':
  import samples, nc0, children, prefix

  z = samples.random(27, 7)
  root = STNode(z)
  good = 0
  bad = 0
  for z in children.subtree(root, -1):
    skip(z)
    if z.skip:
      print(z.sigma)
      bad += 1
    else:
      good += 1
  print('Nodes processed:\t', good)
  print('Nodes skipped:\t', bad)
