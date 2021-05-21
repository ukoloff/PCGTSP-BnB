#
# Enumerate possible children of Search Tree Node
#
from klasses import Task, STNode

def allowed_groups(node: STNode):
  """Построить (достроить) список разрешённых следующих групп для префикса
  """
  sigma = node.sigma
  result = set()
  seen = set()
  if node.parent and node.parent.allowed_groups and all(
      p == q for p, q in zip(node.parent.sigma, node.sigma)):
    sigma = sigma[len(node.parent.sigma):]
    result = node.parent.allowed_groups.copy()
    seen = set(node.parent.sigma)
  for group in sigma:
    seen.add(group)
    for dn in node.task.tree.succ[group]:
      if dn in result:
        continue
      if not all(up in seen for up in node.task.tree.pred[dn]):
        continue
      result.add(dn)
    if group in result:
      result.remove(group)
  node.allowed_groups = result

def children(node: STNode):
  """Генератор узлов дерева - потомков
  """
  if not node.allowed_groups:
    allowed_groups(node)
  for group in node.allowed_groups:
    child = STNode(node.task, node.sigma + (group,))
    child.parent = node
    allowed_groups(child)
    yield child

def subtree(node: STNode):
  """Генерировать полное поддерево поиска
  """
  from queue import Queue
  Q = Queue()
  Q.put(node)
  while not Q.empty():
    x = Q.get()
    for z in children(x):
      Q.put(z)
      yield z

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  # z = samples.load("e1x_1")
  root = STNode(z)
  for z in subtree(root):
    print(z.sigma, z.allowed_groups)
