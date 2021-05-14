#
# Node Condensation Zero
#
# Convert full Town's graph to Cluster's graph
#
import networkx as nx

def nc0(task):
  """(task) -> None
  """
  res = nx.create_empty_copy(task.tree)

  for A in task.tree:
    for B in task.tree:
      if A is B:
        continue
      if B == 1:
        if task.tree.out_degree(A) != 0:
          continue
      else:
        if task.tree_closure.has_edge(B, A):
          continue
        if task.tree_closure.has_edge(A, B) and not task.tree.has_edge(A, B):
          continue
      w = min(w
        for cityA in task.clusters[A]
        for cityB in task.clusters[B]
        if task.dists.has_edge(cityA, cityB)
        for w in [task.dists.edges[cityA, cityB]['weight']]
        if w >= 0
      )
      res.add_edge(A, B, weight=w)
  task.initialNC = res

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  # z = samples.load("e1x_1")
  nc0(z)
  print(z)
