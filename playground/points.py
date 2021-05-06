import random
import networkx as nx

def points():
    """
    Generate points in circle
    Center: 0, 0; Radius: 1
    """
    while True:
        pt = (*[random.uniform(-1, 1) for i in range(2)],)
        if sum(x * x for x in pt) < 1:
            yield pt

def pays(n):
  """
  Generate n towns on the circle
  """
  res = nx.complete_graph(n)
  for node, pt in zip(res, points()):
    # res.add_node(node, pt=pt)
    res.nodes[node]['@'] = pt

  for u, v in res.edges:
    res[u][v]['weight'] = sum((x-y) ** 2 for x, y in zip(res.nodes[u]['@'], res.nodes[v]['@'])) ** 0.5

  return res

if __name__ == '__main__':
    from itertools import islice

    # print(*islice(points(), 10))
    print(*pays(5).edges.data('weight'))
