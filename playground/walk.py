#
# Полный обход дерева решений с отсечением
#
import networkx as nx

from klasses import Task, STNode
import prefix, nc0, nc, cut_prefix, children

def solve(task: Task):
  """Обход дерева решений
  """
  nc0.nc0(task)
  for node in children.subtree(STNode(task)):
    print(node.sigma, end='\t', flush=True)
    cut_prefix.skip(node)
    if node.skip:
      print('!')
      continue
    if node.is_leaf():
      nc.upper_bound(node)
      print(node.bounds)
      continue
    nc.bounds(node)
    print(node.bounds, end='\t', flush=True)
    if node.bounds['LB'] > task.UB:
      node.skip = True
      print('!')
      continue
    print()

if __name__ == '__main__':
  import samples

  solve(samples.random(100, 12))
