import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations

def createModel(model_name, G, tree, first_node_idx = 0):
    with gp.Env(empty = True) as env:
        env.setParam('LogToConsole', 0)
        env.start()
        
        model=gp.Model(model_name, env=env)
        
        n = len(G)
        n_list = list(G)
        n_dict = {n_list[idx]: idx for idx in range(n)}
        first_node = n_list[first_node_idx]

        tree_closure = nx.transitive_closure_dag(tree)
        A = nx.to_numpy_matrix(G, nodelist=list(G))

        x_ij, cost = gp.multidict(dict(np.ndenumerate(A)))
        y_ij, dummy_cost = gp.multidict(dict(np.ndenumerate(np.zeros((n,n)))))

        ### VARIABLES
        x = model.addVars(x_ij, vtype=GRB.BINARY, name='x')
        y = model.addVars(y_ij, name='y')

        ### OBJECTIVE
        objective = model.setObjective(x.prod(cost), GRB.MINIMIZE)

        ### CONSTRAINTS
        ### FLOW CONSERVATION
        for v_i in range(n):
            model.addConstr(sum(x[v_i,v_j] for v_j in range(n) 
                                                if (n_list[v_i],n_list[v_j]) in G.edges) == 1, f'fc_outer_{n_list[v_i]}')

            model.addConstr(sum(x[v_j,v_i] for v_j in range(n) 
                                                if (n_list[v_j],n_list[v_i]) in G.edges) == 1, f'fc_inner_{n_list[v_i]}')
    

        ### SUB-TOUR ELIMINATION
        for e in G.edges:
            i = n_dict[e[0]]
            j = n_dict[e[1]]

            if not first_node_idx in (i,j):               
                model.addConstr(y[i,j] - x[i,j] >= 0, f'ste1_{n_list[i]}_{n_list[j]}')
                model.addConstr(y[i,j] + y[j,i] == 1, f'ste2_{n_list[i]}_{n_list[j]}')

        idxs = list(range(first_node_idx)) + list(range(first_node_idx + 1, n))
        for i,j,k in permutations(idxs,3):
            model.addConstr(y[i,j] + y[j,k] + y[k,i] <= 2, f'ste3_{n_list[i]}_{n_list[j]}_{n_list[k]}')

        ### PRECEDENCE CONSTRAINTS

        for e in tree_closure.edges:
            i = n_dict[e[0]]
            j = n_dict[e[1]]
            
            model.addConstr(y[i,j] == 1, f'pc_{n_list[i]}_{n_list[j]}')

        return model, x

def optimizeModel(model):
    model.setParam(GRB.Param.TimeLimit,1)
    model.optimize()
    
    return model.status, model.objboundc


def main(model_name, G_c, tree):
    # print(G_c.nodes)
    # print(G_c.edges(data='weight'))
    
    # comps = list(nx.strongly_connected_components(G))
    # print(len(comps))
    # exit(1)

    model, x = createModel(model_name, G_c, tree)
    model.write(f'tmp/{model_name}.lp')

    status, LB = optimizeModel(model)
    print(status, LB)
	

if __name__ == '__main__':
    from fromPCGLNS import getInstance
    from salmanize import nc0

    model_name='e5x_1'
    G, clusters, tree = getInstance(f'pcglns/{model_name}.pcglns')
    tree_closure = nx.transitive_closure_dag(tree)

    G_c = nc0(G, clusters, tree, tree_closure)
    main(model_name, G_c, tree)
