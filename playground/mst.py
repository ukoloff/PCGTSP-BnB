#
# Test Minimal Spanning Tree (undirected)
#
import networkx as nx

import points


def mst(graph):
    return nx.minimum_spanning_tree(graph)

if __name__ == '__main__':
  z = points.pays(31)
  w = mst(z)
  with open("tmp/mst.html", "w") as f:
    print(points.v2svg(w, edges=True, earth=True, wrap=True), file=f)
