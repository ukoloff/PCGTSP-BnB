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

def factorial(n):
  result = 1
  for i in range(2, n + 1):
    result *= i
  return result

if __name__ == '__main__':
  import samples, nc0, children, prefix

  task = samples.random(27, 7)

  root = STNode(task)
  total = sum(1 for z in children.subtree(root))
  print("Task:", len(task.dists), "/", len(task.clusters))
  print("Total ->\t", total, "\t// of", factorial(len(task.clusters) - 1))

  for order in (None, +1, -1):
    history.clear()
    root = STNode(task)
    good = 0
    bad = 0
    for z in children.subtree(root, order):
      skip(z)
      if z.skip:
        # print(z.sigma)
        bad += 1
      else:
        good += 1
    print('Order', order, "->\t", good, "\t// skipped:", bad)
