import random
import networkx as nx


def points():
    """
    Generate points in circle
    Center: 0, 0; Radius: 1
    """
    while True:
        # Tuple comprehension ;-)
        pt = tuple(random.uniform(-1, 1) for i in range(2))
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
        res[u][v]['weight'] = sum(
            (x-y) ** 2 for x, y in zip(res.nodes[u]['@'], res.nodes[v]['@'])) ** 0.5

    if directed:
        res = nx.to_directed(res)
    return res


def v2svg(pays, vertices=True, edges=False, earth=False, wrap=False):
    """
    Draw vertices of a graph as SVG
    """
    res = ""
    if earth:
        res = f'<circle class="earth" r="1" cx="0" cy="0"></circle>]\n'
    if vertices:
        for node in pays:
            pt = pays.nodes[node]['@']
            res += f'<circle class="vertex" cx="{pt[0]}" cy="{pt[1]}"></circle>\n'
    if edges:
        for u, v in pays.edges:
            res += f'<line class="edge" x1="{pays.nodes[u]["@"][0]}" y1="{pays.nodes[u]["@"][1]}" x2="{pays.nodes[v]["@"][0]}" y2="{pays.nodes[v]["@"][1]}"></line>\n'
    if wrap:
        from html5 import html5
        res = html5(res)
    return res


if __name__ == '__main__':
    from itertools import islice

    # print(*islice(points(), 10))
    z = pays(5, True)
    print(*pays(5, True).edges.data('weight'))
    print(v2svg(pays(15), earth=True, wrap=True, edges=False),
          file=open("tmp/towns.html", "w"))
