#
# Полный обход дерева решений с отсечением
#
from datetime import timedelta
from timeit import default_timer as timer

from klasses import Task, STNode
import prefix, nc, cut_prefix, children

def solve(task: Task):
  """Обход дерева решений
  """
  nc.initL1(task)
  nc.initL2(task)

  last_len = 0
  last = timer()
  start = last

  total_nodes = 0
  skipped_nodes = 0

  root = STNode(task)
  for node in subtree(root, order=-1):
    total_nodes += 1
    now = timer()
    if now > last + 60:
      print('+', timedelta(seconds=now - start),
        '\tNodes:', total_nodes, f'\tSkipped: {int(skipped_nodes / total_nodes * 100)}%')
      last = now

    if len(node.sigma) != last_len:
      last_len = len(node.sigma)
      nc.history.clear()
      cut_prefix.history.clear()

    print(node.sigma, end='\t', flush=True)
    cut_prefix.skip(node)
    if node.skip:
      skipped_nodes += 1
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
      skipped_nodes += 1
      print('!')
      continue
    updateLB(node)
    print(f'\tLB={root.LB} // {(root.task.UB - root.LB) / root.task.UB * 100:.0f}%')


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
    if x.parent:
      x.parent.pending.remove(x)
    yield x
    if x.skip:
      continue

    x.pending = set()
    x.children = set()
    seq = children.children(x)
    if order is not None:
      seq = sorted(seq, key=lambda x: len(x.allowed_groups), reverse=order > 0)
    for z in seq:
      x.pending.add(z)
      Q.put(z)


def updateLB(node: STNode):
  node.LB = node.bounds['LB']
  if node.parent:
    node.parent.children.add(node)
    if len(node.parent.pending) > 0:
      return
  while node.parent:
    node = node.parent
    oldLB = node.LB
    newLB = min((z.LB for z in node.children), default=oldLB)
    if oldLB >= newLB:
      return
    node.LB = newLB


if __name__ == '__main__':
  import sys

  import samples

  src = "e5x_1" if len(sys.argv) < 2 else sys.argv[1]
  print("Loading:", src)

  # task = samples.random(1000, 12)
  task = samples.load(src)

  print("Size:\t", task.dists.order(), '/', len(task.clusters))
  print("UB:\t", task.UB)
  print("Solution so far:\t", task.solution)
  print()

  solve(task)
