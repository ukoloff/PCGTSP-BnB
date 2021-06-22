#! /bin/python

from gurobipy import *
import modelextender as me
import numpy as np

def createModel(model_name, n, cn, mat, clusters, ass, x_start, D_start, precedence):
    model=Model(model_name)
    x_ij, cost=multidict(dict(np.ndenumerate(mat)))
    DD, dummy_cost=multidict(dict(np.ndenumerate(np.zeros((cn,cn)))))
    UU = DD.copy()

	#
	#	variables: 
	#		x - arc using indicator 			(n x n)  	- BINARY
	#       U - cluster transition indicators	(cn x cn)	
	#		D - cluster precedence				(cn x cn)
	#
	#
    x=model.addVars(x_ij, vtype=GRB.BINARY, name='x')
    U=model.addVars(UU, name='U')
    D=model.addVars(DD, name='D')

	#
    #	CONSTRAINTS
    #
    NUMBER_OF_NODES=n
    LAST_CLUSTER= cn - 1
    NUMBER_OF_CLUSTERS = cn
    FPIC=clusters[1][0]                 # FIRST_POINT_FROM_INTERMEDIATE_CLUSTER
    FPLC=clusters[LAST_CLUSTER][0]      # FIRST_POINT_FROM_LAST_CLUSTER

    #
    #	relation between x_ij and U
    #
    x_ij2U=[]
    for idx1 in range(NUMBER_OF_CLUSTERS):
    	for idx2 in range(NUMBER_OF_CLUSTERS):
            x_ij2U.append(model.addConstr(U[idx1,idx2] == sum(x[i,j] for i in clusters[idx1] for j in clusters[idx2]),
                'x_ij-2-U-({0},{1})'.format(idx1,idx2)))

    #
    #	flow constraints
    #	each node from any cluster
    #	should have the same active inside and outside arcs (equal implicitly to 0 or 1)
    #
    flow=[]
    for inode in range(NUMBER_OF_NODES):
        flow.append(model.addConstr(sum(x[i,inode] for i in range(NUMBER_OF_NODES) if i!=inode) 
            ==  sum(x[inode,j] for j in range(NUMBER_OF_NODES) if j != inode ), 
        	'flow-{0}'.format(inode))) 

    #
    #	Each cluster 
    #	has exactly one arc going outside
    #
    
    outside=[]
    for idx1 in range(NUMBER_OF_CLUSTERS):
        others = [idx for idx in range(NUMBER_OF_CLUSTERS) if idx != idx1]
        outside.append(model.addConstr(sum(U[idx1,idx2] for idx2 in others) == 1,'outside-{0}'.format(idx1)))

    #
    #	Each cluster 
    #	has exactly one arc going inside
    #
    
    inside=[]
    for idx1 in range(NUMBER_OF_CLUSTERS):
        others = [idx for idx in range(NUMBER_OF_CLUSTERS) if idx != idx1]
        inside.append(model.addConstr(sum(U[idx2,idx1] for idx2 in others) == 1,'inside-{0}'.format(idx1)))

    #
    #	No points inside the same cluster can be connected
    #
    
    inner = model.addConstr(sum(U[idx,idx] for idx in range(NUMBER_OF_CLUSTERS)) == 0, 'internal')

    #      
    # 	no direct way to the first cluster and from the last one
    #	COMMENTED - NO NEED
    

    # first=model.addConstr(sum(U[idx,0] for idx in range(1, NUMBER_OF_CLUSTERS)) == 0, 'first-STOP')
    # last=model.addConstr(sum(U[LAST_CLUSTER,idx] for idx in range(LAST_CLUSTER)) == 0, 'last-STOP')

    #
    #	PRECEDENCE CONSTRAINTS
    #
    
    #
    #	An assignment
    #	Each cluster except the first one can be preceeded by some other clusters
    #
    preceeds=[]
    for idx1 in range(NUMBER_OF_CLUSTERS):
        if idx1 in precedence:
            for idx2 in precedence[idx1]:
                if idx2 != 0:
                    preceeds.append(model.addConstr(D[idx1,idx2] == 1, 'prec-({0},{1})'.format(idx1,idx2)))

    #
    #	Asymmetry
    #
    asymmetry=[]
    for idx1 in range(1, NUMBER_OF_CLUSTERS):
    	for idx2 in range(idx1 + 1, NUMBER_OF_CLUSTERS):
    		asymmetry.append(model.addConstr(D[idx1,idx2] + D[idx2,idx1] == 1, 'asym-({0},{1})'.format(idx1,idx2)))

    #
    #	Bounds
    #
    bounds=[]
    for idx1 in range(1,NUMBER_OF_CLUSTERS):
    	for idx2 in range(1,NUMBER_OF_CLUSTERS):
    		if idx2 != idx1:
    			bounds.append(model.addConstr(U[idx1,idx2] - D[idx1,idx2] <= 0, 'bound-({0},{1})'.format(idx1,idx2)))

    #
    #	triangle constraint
    #
    triangle=[]
    for idx1 in range(1,NUMBER_OF_CLUSTERS):
    	for idx2 in range(1,NUMBER_OF_CLUSTERS):
    		if idx2 != idx1:
    			for idx3 in range(1,NUMBER_OF_CLUSTERS):
    				if (idx3 != idx2) and (idx3 != idx1):
    					triangle.append(model.addConstr(U[idx1,idx2] + D[idx2,idx3] + U[idx3,idx2] + D[idx3,idx1] + U[idx1,idx3] <= 2,
                            'trio-({0}-{1}-{2})'.format(idx1,idx2,idx3)))

    #
    #   additional preceedence constraint
    #   if cluster p preceeds cluster q and q preceeds cluster r, then U_pq=0
    #
    APC=[]
    for idx in range(1,NUMBER_OF_CLUSTERS):
        if idx in precedence:
            for jdx in precedence[idx]:
                if jdx in precedence:
                    if jdx != 0:
                        for kdx in precedence[jdx]:
                            if  kdx != 0:
                                APC.append(model.addConstr(U[idx,kdx]==0, 'APC-({0}-{1})'.format(idx,kdx)))

    #
    #	HEURISTIC
    #
    for i in range(n):
    	for j in range(n):
    		x[i,j].start = x_start[i,j]

    for idx1 in range(NUMBER_OF_CLUSTERS):
    	for idx2 in range(NUMBER_OF_CLUSTERS):
    		if D_start[idx1,idx2]>0:
    			D[idx1,idx2].start = D_start[idx1,idx2]


    #
    #	OBJECTIVE FUNCTION
    #
    objective=model.setObjective(x.prod(cost), GRB.MINIMIZE)

    #
    # 	PARAMETERS
    #
    # 	brach rule: Up first!

    # model.setParam(GRB.Param.BranchDir, 1)
    model.setParam(GRB.Param.Threads,12)
    return (model, x)

#
#	write model
#

def writeModel(fname, model):
	model.write(fname)



#
#
#	OK! Let's OPTIMIZE now!
#
#

def optimize(n, model, X):
	model.optimize()

	x = np.zeros((n,n), dtype=int)
	for i in range(n):
		for j in range(n):
			x[i,j] = X[i,j].x

	return x


    #
    #
    #	UNIT TESTS
    #
    #

def test():
    n=6
    cn=3
    clust=[[0,1], [2], [3,4,5]]
    mat=np.array(range(36)).reshape(6,6)
    ass=[0,0,1,2,2,2]
    heu=[1, 4, 2]
    prec={2: [1]}
    mat[clust[1],clust[2]] = -1
    print('Test1\nGurobi model self-test\nInput values\n============')
    print(n, cn)
    print(mat)
    print(clust)
    print(ass)
    print(heu)
    print(prec)

    n, cn, mat, clust, ass, heu, prec = me.adjustModel(n, cn, mat, clust, ass, heu, prec)
    print('Adjusted values\n============')
    print(n, cn)
    print(mat)
    print(clust)
    print(ass)
    print(heu)
    print(prec)
    print('============')

    x_start, D_start = me.prepareInitialSolution(n, cn, heu, ass)
    
    model_name='TEST-MODEL'
    model, xVar = createModel(model_name, n, cn, mat, clust, ass, x_start, D_start, prec) 
    model.write(model_name + '.lp')

    print('Model is written to {0}.lp'.format(model_name))
    print('Try to find an optimim solution ...')

    x=optimize(n, model, xVar)

    print(x)


if __name__ == '__main__':
	test()

