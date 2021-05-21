#
# Enumerate possible children of Search Tree Node
#
from klasses import *

def allowed_groups(node: STNode):
  sigma = node.sigma
  result = set()
  seen = set()
  if node.parent and node.parent.allowed_groups and node.parent.sigma.startswith(node.sigma):
    sigma = sigma[len(node.parent.sigma):]
    result = node.parent.allowed_groups
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

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  root = STNode(z)
  allowed_groups(root)
