#
# Построение модели для Gurobi
# Альтернативная версия
#
import re
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

# Force banner
gp.Model('-')


def run(model: gp.Model):
    model.Params.TimeLimit = 1.01
    model.optimize()
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
      raise RuntimeError(f"Gurobi exited with status {model.status}")
    return model.ObjBound


def model(graph: nx.DiGraph, tree_closure: nx.DiGraph, start_node=1):
    m = gp.Model('PC-ATSPxy')
    m.Params.LogToConsole = 0
    m.Params.Threads = 1
    # m.Params.TimeLimit = 0.01

    vertices = [start_node] + [v for v in graph if v != start_node]
    n = len(vertices)
    iVert = {v: i for i, v in enumerate(vertices)}

    Xs, costs = gp.multidict(
        ((iVert[u], iVert[v]), w)
        for u, v, w in graph.edges.data('weight') if u != v)
    Ys = gp.tuplelist(
        (u, v)
        for u in range(1, n) for v in range(1, n) if u != v)

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
    run(m)
    print('Result:', m.ObjBoundC)
    print('Time:', m.Runtime)
    print('Status:', m.Status)

    tour = nx.DiGraph(
      [int(v) for v in re.findall(r"\d+", v.VarName)]
      for v in m.getVars()
      if v.VarName.startswith('x[') and v.X != 0
    )
    print("Tour:", *nx.simple_cycles(tour))

    # Time it!
    build = timeit(lambda: model(graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: model(
        graph, task.tree_closure).optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')
