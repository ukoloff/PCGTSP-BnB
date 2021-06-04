#
# Noon-Bean transformation
#
import networkx as nx

from klasses import Task, STNode


def noon_bean(node: STNode):
    """Построить Noon-Bean для суффикса
    """
    result = nx.DiGraph()

    nc0 = node.task.initialNC
    clusters = node.task.clusters
    pt2c = {pt: (g, i) for g, pts in clusters.items()
            for i, pt in enumerate(pts)}
    omit = set(node.sigma[1:-1])
    for u, v, w in node.task.dists.edges.data('weight'):
        ug, ui = pt2c[u]
        if ug in omit:
            continue
        vg, vi = pt2c[v]
        if vg in omit:
            continue
        if ug == vg:
            continue
        if vg == node.sigma[0] and ug == node.sigma[-1]:
            continue
        if len(node.sigma) > 1:
            if ug == node.sigma[0] or vg == node.sigma[-1]:
                continue
        if not nc0.has_edge(ug, vg):
            continue
        result.add_edge(clusters[ug][(ui - 1) %
                                     len(clusters[ug])], v, weight=w)

    # Build 0-cycles
    if len(node.sigma) > 1:
        # Avoid cycles @ prefix groups
        # omit = set(node.sigma)
        pass

    result.add_weighted_edges_from(
        (pts[j], pts[(j + 1) % len(pts)], 0)
        for i, pts in clusters.items()
        if not i in omit
        if len(pts) > 1
        for j in range(len(pts))
    )

    # Add 0-path thru prefix
    if len(node.sigma) > 1:
        # if len(clusters[node.sigma[0]]) == 1:
        #     result.add_weighted_edges_from(
        #         (clusters[node.sigma[0]][0], v, 0) for v in clusters[node.sigma[-1]])
        # elif len(clusters[node.sigma[-1]]) == 1:
        #     result.add_weighted_edges_from(
        #         (v, clusters[node.sigma[-1]][0], 0) for v in clusters[node.sigma[0]])
        # else:
        #     result.add_weighted_edges_from((v, '!', 0)
        #                                    for v in clusters[node.sigma[0]])
        #     result.add_weighted_edges_from(('!', v, 0)
        #                                    for v in clusters[node.sigma[-1]])
        result.add_weighted_edges_from(
            (u, v, 0) for u in clusters[node.sigma[0]] for v in clusters[node.sigma[-1]])

    return result


if __name__ == '__main__':
    import samples
    import nc0
    import children

    task = samples.load("e5x_1")
    nc0.nc0(task)
    root = STNode(task)
    for node in children.subtree(root):
        nb = noon_bean(node)
        print(node.sigma, len(nb))
