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

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  root = STNode(z)
  w = [*children(root)]
