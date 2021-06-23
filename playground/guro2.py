#
# Построение модели для Gurobi
# Альтернативная версия
#
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB


def model(graph: nx.DiGraph, tree_closure: nx.DiGraph, start_node=1):
    m = gp.Model('PCATSPxy')
    m.Params.LogToConsole = 0

    vertices = list(graph)
    n = len(vertices)
    iVert = {v: i for i, v in enumerate(vertices)}
    start_idx = iVert[start_node]

    Xs, costs = gp.multidict(
        ((iVert[u], iVert[v]), w)
        for u, v, w in graph.edges.data('weight') if u != v)
    Ys = gp.tuplelist(
        (u, v)
        for u in range(n) if u != start_idx for v in range(n) if v != start_idx if u != v)

    # VARIABLES
    x = m.addVars(Xs, vtype=GRB.BINARY, name='x')
    y = m.addVars(Ys, name='y')

    # OBJECTIVE
    m.setObjective(x.prod(costs), GRB.MINIMIZE)

    # CONSTRAINTS
    # FLOW CONSERVATION
    m.addConstrs((x.sum(i, '*') == 1 for i in range(n)), 'out')
    m.addConstrs((x.sum('*', i) == 1 for i in range(n)), 'in')

    # SUB-TOUR ELIMINATION
    m.addConstrs(
        (y[u, v] - x[u, v] >= 0
            for u, v in x if u != start_idx if v != start_idx),
        'xy')
    m.addConstrs(
        (y[u, v] + y[v, u] == 1
            for u in range(n) if u != start_idx for v in range(u + 1, n) if v != start_idx),
        'yy')
    m.addConstrs(
        (y[a, b] + y[b, c] + y[c, a] <= 2
            for a, b, c in (
            (iVert[u], iVert[v], iVert[w])
            for u, v in graph.edges
            if u != start_node and v != start_node
            if iVert[u] < iVert[v]
            for w in set(graph.predecessors(u)) & set(graph.successors(v))
            if w != start_node
            if iVert[u] < iVert[w])),
        'tri')

    # PRECEDENCE CONSTRAINTS
    m.addConstrs(
        (y[u, v] == 1
         for u, v in (
            (iVert[u], iVert[v])
            for u, v in tree_closure.edges
            if u in iVert and v in iVert)
         if u != start_idx and v != start_idx
         ),
        'pc')

    return m


if __name__ == '__main__':
    from timeit import timeit
    import samples
    import nc
    from klasses import Task, STNode

    # z = samples.random(27, 7)
    task = samples.load("e5x_1")
    nc.initL1(task)
    root = STNode(task)
    graph = nc.nc(root, L=1)
    # tree = nc.get_order(root, graph)

    m = model(graph, task.tree_closure)
    # m.write('!!!.lp')
    m.optimize()
    print('Result:', m.objboundc)
    for v in m.getVars():
        if not v.VarName.startswith('x[') or v.X == 0:
            continue
        print(v.VarName[2:-1], end=' ')

    # Time it!
    print()
    build = timeit(lambda: model(graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: model(
        graph, task.tree_closure).optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')

    from guro2x import create_model
    print('[guro2x]')
    build = timeit(lambda: create_model(
        graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: create_model(graph, task.tree_closure)[
                   0].optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')

    import guro2z
    print('[guro2z]')
    build = timeit(lambda: guro2z.model(graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: guro2z.model(graph, task.tree_closure).optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')
