#
# Построение модели для Gurobi
# Массовое добавление ограничений
#
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def model(G, T_closure, first_node=1):
    W = {e: G[e[0]][e[1]]['weight'] for e in G.edges}
    x_ij, cost = gp.multidict(W)
    CP = {(u, v): None for u in G for v in G if u != v}
    y_ij, _ = gp.multidict(CP)

    v_dict = {v: i for i, v in enumerate(list(G.nodes))}

    model = gp.Model('tmp')
    model.Params.LogToConsole = False

    # VARIABLES
    x = model.addVars(x_ij, vtype=GRB.BINARY, name='x')
    y = model.addVars(y_ij,  name='y')

    # OBJECTIVE
    model.setObjective(x.prod(cost), GRB.MINIMIZE)

    # CONSTRAINTS
    # FLOW CONSERVATION
    model.addConstrs((x.sum(v, '*') == 1 for v in G), name='fc_outer')
    model.addConstrs((x.sum('*', v) == 1 for v in G), name='fc_inner')

    # SUBTOUR ELIMINATION
    model.addConstrs((x[u, v] <= y[u, v]
                      for u, v in G.edges if not first_node in (u, v)), name='ste1')
    model.addConstrs((y[u, v] + y[v, u] == 1 for u in G for v in G if u !=
                      v and not first_node in (u, v)), name='ste2')
    model.addConstrs((y[u, v] + y[v, w] + y[w, u] <= 2
                      for u in G for v in G
                      for w in nx.descendants(G, v) & nx.ancestors(G, u)
                      if v_dict[u] < v_dict[v] and v_dict[u] < v_dict[w] and not first_node in (u, v, w)), name='ste3')

    # PRECEDENCE CONSTRAINTS
    model.addConstrs((y[u, v] == 1 for u, v in T_closure.edges), name='pc')

    return model
