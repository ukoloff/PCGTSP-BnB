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

def pays(n, directed=False):
  """
  Generate n towns on the circle
  """
  res = nx.complete_graph(n)
  for node, pt in zip(res, points()):
    # res.add_node(node, pt=pt)
    res.nodes[node]['@'] = pt

  for u, v in res.edges:
    res[u][v]['weight'] = sum((x-y) ** 2 for x, y in zip(res.nodes[u]['@'], res.nodes[v]['@'])) ** 0.5

  if directed:
    res = nx.to_directed(res)
  return res

def v2svg(pays, wrap=False):
  """
  Draw vertices of a graph as SVG
  """
  res = ""
  for node in pays:
    pt = pays.nodes[node]['@']
    res += f'<circle class="town" cx="{pt[0]}" cy="{pt[1]}"></circle>'
  if wrap:
    from  html5 import html5
    res = html5(res)
  return res

if __name__ == '__main__':
    from itertools import islice

    # print(*islice(points(), 10))
    z = pays(5, True)
    print(*pays(5, True).edges.data('weight'))
    print(v2svg(pays(15), True), file=open("x.html", "w"))
