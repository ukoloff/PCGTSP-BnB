#
# Test Minimal Spanning Arborescence (directed)
#
import networkx as nx

import points

def msap(graph):
  msap = nx.minimum_spanning_arborescence(graph, preserve_attrs=True)
  res = nx.DiGraph()
  res.add_nodes_from(graph.nodes)
  for node in graph.nodes:
    for k, v in graph.nodes[node].items():
      res.nodes[node][k] = v
  res.add_edges_from(msap.edges)
  return res

if __name__ == '__main__':
    z = nx.to_directed(points.pays(31))
    w = msap(z)
    # print(*w.edges)
    with open("tmp/msap.html", "w") as f:
        print(points.v2svg(w, edges=True, earth=True, wrap=True), file=f)
