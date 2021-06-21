import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations

def create_model(model_name, G, tree, first_node_idx = 0):
    model = gp.Model(model_name)
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
    fc_inner=[]
    fc_outer=[]
    for v_i in range(n):
        fc_outer.append(model.addConstr(sum(x[v_i,v_j] for v_j in range(n) 
                                            if (n_list[v_i],n_list[v_j]) in G.edges) == 1, f'fc_outer_{n_list[v_i]}'))
        
        fc_inner.append(model.addConstr(sum(x[v_j,v_i] for v_j in range(n) 
                                            if (n_list[v_j],n_list[v_i]) in G.edges) == 1, f'fc_inner_{n_list[v_i]}'))
        
    ### SUB-TOUR ELIMINATION
    ste_1=[]
    ste_2=[]
    ste_3=[]
    
    for e in G.edges:
        i = n_dict[e[0]]
        j = n_dict[e[1]]
        
        if not first_node_idx in (i,j):
            ste_1.append(model.addConstr(y[i,j] - x[i,j] >= 0, f'ste1_{n_list[i]}{n_list[j]}'))
            ste_2.append(model.addConstr(y[i,j] + x[j,i] == 1, f'ste2_{n_list[i]}{n_list[j]}'))
        
    perms=set()
        
    for e1,e2,e3 in [p for p in permutations([e for e in G.edges if not first_node in e],3) if sorted(p) == p]:
        if e1[1] == e2[0] and e2[1] == e3[0] and e3[1] == e1[0]:
            perms.add((e1, e2, e3))

    for e1, e2, e3 in perms:
        i = n_dict[e1[0]]
        j = n_dict[e2[0]]
        k = n_dict[e3[0]]
        ste_3.append(model.addConstr(y[i,j] + y[j,k] + y[k,i] <= 2, f'ste3_{n_list[i]}{n_list[j]}{n_list[k]}'))
        
    ### PRECEDENCE CONSTRAINTS
    
    pc = []
    for e in tree_closure.edges:
        i = n_dict[e[0]]
        j = n_dict[e[1]]
                
        pc.append(model.addConstr(y[i,j] == 1, f'pc_{n_list[i]}{n_list[j]}'))

    return model, x


def optimizeModel(model):
    model.setParam(GRB.Param.TimeLimit,1)
    model.optimize()
    
    return model.status, model.objboundc


def main():
	pass

if __name__ == '__main__':
	main()
