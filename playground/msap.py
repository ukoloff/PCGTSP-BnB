#
# Test Minimal Spanning Arborescence (directed)
#
import networkx as nx

import points

def msap(graph):
  msap = nx.minimum_spanning_arborescence(graph, preserve_attrs=True)
  for node in graph.nodes:
    for k, v in graph.nodes[node].items():
      msap.nodes[node][k] = v
  return msap

if __name__ == '__main__':
    z = points.pays(31).to_directed()
    w = msap(z)
    # print(*w.edges)
    with open("tmp/msap.html", "w") as f:
        print(points.v2svg(w, edges=True, earth=True, wrap=True), file=f)
