#
# Node Condensation Zero
#
# Convert full Town's graph to Cluster's graph
#
import networkx as nx

def nc0(dists, clusters, tree):
  """(dist_graph, clustering, tree) -> condensation_graph
  """
  red = nx.transitive_reduction(tree)
  clo = nx.transitive_closure_dag(tree)
  res = nx.create_empty_copy(tree)

  for A in tree:
    for B in tree:
      if A is B:
        continue
      if B == 1:
        if red.out_degree(A) != 0:
          continue
      else:
        if red.has_edge(B, A):
          continue
        if clo.has_edge(A, B) and not red.has_edge(A, B):
          continue
      w = min(w
        for cityA in clusters[A]
        for cityB in clusters[B]
        if dists.has_edge(cityA, cityB)
        for w in [dists.edges[cityA, cityB]['weight']]
        if w >= 0
      )
      res.add_edge(A, B, weight=w)
  return res

if __name__ == '__main__':
  import samples

  # z = samples.random(27, 7)
  z = samples.load("e1x_1")
  w = nc0(*z)
  print(w)
