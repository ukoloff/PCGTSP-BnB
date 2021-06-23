#
# Построение модели для Gurobi
# Альтернативная версия
#
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

# Force banner
gp.Model('-')

def model(graph: nx.DiGraph, tree_closure: nx.DiGraph, start_node=1):
    m = gp.Model('PC-ATSPxy')
    m.Params.LogToConsole = 0
    m.Params.Threads = 1
    # m.Params.TimeLimit = 0.01

    Xs, costs = gp.multidict(
        ((u, v), w)
        for u, v, w in graph.edges.data('weight') if u != v)
    yIndex = [v for v in graph if v != start_node]
    Ys = gp.tuplelist(
        (u, v)
        for u in yIndex for v in yIndex if u != v)

    # VARIABLES
    x = m.addVars(Xs, vtype=GRB.BINARY, name='x')
    y = m.addVars(Ys, name='y')

    # OBJECTIVE
    m.setObjective(x.prod(costs), GRB.MINIMIZE)

    # CONSTRAINTS
    # FLOW CONSERVATION
    for v in graph:
        m.addConstr(x.sum(v, '*') == 1, f'out[{v}]')
        m.addConstr(x.sum('*', v) == 1, f'in[{v}]')

    # SUB-TOUR ELIMINATION
    for u, v in x:
        if u == start_node or v == start_node:
            continue
        m.addConstr(
            y[u, v] >= x[u, v],
            f'xy[{u},{v}]')

    for ia, a in enumerate(yIndex):
        for ib in range(ia + 1, len(yIndex)):
            b = yIndex[ib]
            m.addConstr(
                y[a, b] + y[b, a] == 1,
                f'yy[{a},{b}]')

            for ic in range(ia + 1, len(yIndex)):
                if ib == ic:
                    continue
                c = yIndex[ic]
                m.addConstr(
                    y[a, b] + y[b, c] + y[c, a] <= 2,
                    f'tri[{a},{b},{c}]')

    # PRECEDENCE CONSTRAINTS
    for u, v in tree_closure.edges:
        if u == start_node or v == start_node:
            continue
        if u not in graph or v not in graph:
            continue
        m.addConstr(
            y[u, v] == 1,
            f'pc[{u},{v}]')

    return m


if __name__ == '__main__':
    from timeit import timeit
    import samples
    import nc
    from klasses import Task, STNode

    # z = samples.random(27, 7)
    task = samples.load("34")
    nc.initL1(task)
    root = STNode(task)
    graph = nc.nc(root, L=1)
    # tree = nc.get_order(root, graph)

    m = model(graph, task.tree_closure)
    # m.write('!!!.lp')
    m.optimize()
    print('Result:', m.ObjBoundC)
    print('Time:', m.Runtime)
    print('Status:', m.Status)
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

    # from guro2x import create_model
    # print('[guro2x]')
    # build = timeit(lambda: create_model(
    #     graph, task.tree_closure), number=10) / 10
    # print(f'Build: {build * 1000:.3f}ms')

    # solve = timeit(lambda: create_model(graph, task.tree_closure)[
    #                0].optimize(), number=10) / 10
    # print(f'Build + Solve: {solve * 1000:.3f}ms')

    # import guro2z
    # print('[guro2z]')
    # build = timeit(lambda: guro2z.model(
    #     graph, task.tree_closure), number=10) / 10
    # print(f'Build: {build * 1000:.3f}ms')

    # solve = timeit(lambda: guro2z.model(
    #     graph, task.tree_closure).optimize(), number=10) / 10
    # print(f'Build + Solve: {solve * 1000:.3f}ms')
