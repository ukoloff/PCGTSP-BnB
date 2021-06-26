#
# Полный обход дерева решений с отсечением
#
import sys
from datetime import timedelta
from timeit import default_timer as timer

from klasses import Task, STNode
import prefix, nc, cut_prefix, children, logging, ilayer

def solve(task: Task, log2=sys.stdout):
  """Обход дерева решений
  """
  nc.initL1(task)
  nc.initL2(task)

  last_len = 0
  solve_start = timer()

  root = STNode(task)
  for node in subtree(root, order=-1):
    if len(node.sigma) != last_len:
      if last_len > 0:
        this_layer.dump(root.LB)
      last_len = len(node.sigma)
      nc.history.clear()
      cut_prefix.history.clear()
      this_layer = ilayer.iLayer(last_len)

    print(node.sigma, end='\t', file=log2)
    cut_prefix.skip(node)
    if node.skip:
      this_layer.skipped()
      print('!', file=log2)
      continue
    if node.is_leaf():
      nc.upper_bound(node)
      print(node.bounds, file=log2)
      this_layer.found()
      this_layer.registerUB(node.bounds['UB'], node.sigma)
      continue
    nc.bounds(node)
    print(node.bounds, end='\t', file=log2)
    if node.bounds['LB'] > task.UB:
      node.skip = True
      this_layer.skipped()
      print('!', file=log2)
      continue
    updateLB(node)
    this_layer.registerLB(node.bounds['LB'])
    print(f'\tLB={root.LB} // {(root.task.UB - root.LB) / root.task.UB * 100:.0f}%', file=log2)

  solve_end = timer()
  this_layer.printUB()
  print(f'Elapsed: {solve_end - solve_start:.1f} s')

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

  src = "e3x_2" if len(sys.argv) < 2 else sys.argv[1]
  log2 = logging.start(src)
  print("Loading:", src)

  # task = samples.random(1000, 12)
  task = samples.load(src)

  print("Size:\t", task.dists.order(), '/', len(task.clusters))
  print("UB:\t", task.UB)
  print("Solution so far:\t", task.solution)
  print()

  solve(task, log2)
