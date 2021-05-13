#
# (Undirected) Assignment Problem
#
import networkx as nx

import points


def ap(graph):
    bi = nx.Graph()
    for u, v, w in graph.edges.data('weight'):
        # print(x)
        bi.add_edge((1, u), (2, v), weight=-w)
        bi.add_edge((2, u), (1, v), weight=-w)
    print(bi)
    edges = nx.min_edge_cover(bi)
    # print(edges)
    res = nx.create_empty_copy(graph)
    for (p, u), (q, v) in edges:
      res.add_edge(u, v)
    return res


if __name__ == '__main__':
    z = points.pays(7)
    w = ap(z)
    with open("tmp/ap.html", "w") as f:
        print(points.v2svg(w, edges=True, earth=True, wrap=True), file=f)
