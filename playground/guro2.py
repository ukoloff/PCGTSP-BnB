#
# Построение модели для Gurobi
# Альтернативная версия
#
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def model(graph: nx.DiGraph, tree_closure: nx.DiGraph, start_node=1):
    m = gp.Model('pctsp')
    m.Params.LogToConsole = False

    vertices = list(graph)
    n = len(vertices)
    iVert = {v: i for i, v in enumerate(vertices)}
    start_idx = iVert[start_node]

    Xs, costs = gp.multidict(np.ndenumerate(nx.to_numpy_matrix(graph)))
    Ys, _ = gp.multidict(np.ndenumerate(np.zeros((n, n))))

    # VARIABLES
    x = m.addVars(Xs, vtype=GRB.BINARY, name='x')
    y = m.addVars(Ys, name='y')

    # OBJECTIVE
    objective = m.setObjective(x.prod(costs), GRB.MINIMIZE)

    # CONSTRAINTS
    # FLOW CONSERVATION
    for v in graph:
        m.addConstr(
            sum(x[iVert[v], iVert[w]] for w in graph.successors(v)) == 1,
            f'out_{iVert[v]}')
        m.addConstr(
            sum(x[iVert[w], iVert[v]] for w in graph.predecessors(v)) == 1,
            f'in_{iVert[v]}')

    # SUB-TOUR ELIMINATION
    for u, v in graph.edges:
        iu = iVert[u]
        iv = iVert[v]

        if start_idx in (iu, iv):
            continue

        m.addConstr(
            y[iu, iv] - x[iu, iv] >= 0,
            f'yx_{iu}_{iv}')
        m.addConstr(
            y[iu, iv] + y[iv, iu] == 1,
            f'yy_{iu}_{iv}'
        )

        if iu >= iv:
            continue
        for w in set(graph.predecessors(u)) & set(graph.successors(v)):
            iw = iVert[w]
            if iw == start_idx:
                continue
            if iu >= iw:
                continue
            m.addConstr(
                y[iu, iv] + y[iv, iw] + y[iw, iu] <= 2,
                f'tri_{iu}_{iv}_{iw}')

    # PRECEDENCE CONSTRAINTS
    for u, v in tree_closure.edges:
        if u not in iVert or v not in iVert:
          continue
        iu = iVert[u]
        iv = iVert[v]
        m.addConstr(
            y[iu, iv] == 1,
            f'pc_{iu}_{iv}')

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
        print(v.VarName[2:-1], end='\t')

    # Time it!
    print()
    build = timeit(lambda: model(graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: model(graph, task.tree_closure).optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')

    from guro2x import create_model
    print('[guro2x]')
    build = timeit(lambda: create_model(graph, task.tree_closure), number=10) / 10
    print(f'Build: {build * 1000:.3f}ms')

    solve = timeit(lambda: create_model(graph, task.tree_closure)[0].optimize(), number=10) / 10
    print(f'Build + Solve: {solve * 1000:.3f}ms')
