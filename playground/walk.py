#
# Полный обход дерева решений с отсечением
#
from datetime import timedelta
from timeit import default_timer as timer

from klasses import Task, STNode
import prefix, nc0, nc, cut_prefix, children

def solve(task: Task):
  """Обход дерева решений
  """
  nc0.nc0(task)

  last = timer()
  start = last

  for node in subtree(STNode(task), order=-1):
    now = timer()
    if now > last + 60:
      print('+', timedelta(seconds=now - start))
      last = now

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


def subtree(node: STNode, order=None):
  """Генерировать полное поддерево поиска
  order:
    - None: as is
    - +1: Large successor set first
    - -1: Large successor set last
  """
  from queue import Queue

  Q = Queue()
  Q.put(node)
  # if not node.allowed_groups:
  #   children.allowed_groups(node)
  while not Q.empty():
    x = Q.get()
    x.skip = False
    yield x
    if x.skip:
      continue
    seq = children.children(x)
    if order is not None:
      seq = sorted(seq, key=lambda x: len(x.allowed_groups), reverse=order > 0)
    for z in seq:
      Q.put(z)


if __name__ == '__main__':
  import samples

  # task = samples.random(1000, 12)
  task = samples.load("e1x_1")

  solve(task)
