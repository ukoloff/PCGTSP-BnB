#
# Node Condensation Zero
#
# Convert full Town's graph to Cluster's graph
#
import networkx as nx

from klasses import Task

def nc0(task: Task, start_cluster=1):
  """Рассчитать минимальные расстояния между группами с учётом PC
  """
  res = nx.DiGraph()

  for A in task.tree:
    for B in task.tree:
      if A is B:
        continue
      if B == start_cluster:
        if task.tree.out_degree(A) != 0:
          continue
      else:
        if task.tree_closure.has_edge(B, A):
          continue
        if task.tree_closure.has_edge(A, B) and not task.tree.has_edge(A, B):
          continue
      w = min(w for w in
          (task.dists.edges[cityA, cityB]['weight']
            for cityA in task.clusters[A]
            for cityB in task.clusters[B]
            if task.dists.has_edge(cityA, cityB))
          if w >= 0)
      res.add_edge(A, B, weight=w)
  task.initialNC = res

if __name__ == '__main__':
  import samples

  z = samples.random(27, 7)
  # z = samples.load("e1x_1")
  nc0(z)
  print(z)
