import networkx as nx
import sys

def createVariables(G):
    vars = {(int(e[0]), int(e[1])): e[2] for e in G.edges(data='weight')}
    return vars

def formatVar(e, prefix = 'x'):
    return f'{prefix}{e[0]}_{e[1]}'  

def prepareObjective(vars):
    chunks = [f'+ {w} {formatVar(e)}' if w >= 0 else f'- {w} {formatVar(e)}' for e, w in vars.items() if abs(w) > 0]
    chunks[0] = chunks[0][2:]
    obj = ' '.join(chunks)
    return obj
   
def prepareAPconstraints(G, vars):
    cons1 = []
    cons2 = []
    for v in G.nodes:
        iv = int(v)
        con1 = ' + '.join([f'{formatVar(e)}' for e in vars if e[1] == iv])
        con2 = ' + '.join([f'{formatVar(e)}' for e in vars if e[0] == iv])
        if con1:
            cons1.append(con1 + ' = 1')
        if con2:
            cons2.append(con2 + ' = 1')
        
    return cons1, cons2       
   
def prepareSTEconstraints(v1, vars):
    v1 = int(v1)
    cons3 = [f'{formatVar(e,"y")} - {formatVar(e)} >= 0' for e in vars if not v1 in e]
    cons4 = [f'{formatVar(e,"y")} + {formatVar(e[::-1],"y")} = 1' for e in vars if not v1 in e]
    
    cons5 = []
    for eij in vars:
        if not v1 in eij:
            for ejk in vars:
                if (not v1 in ejk) and (eij[1] == ejk[0]):
                    for eki in vars:
                        if (not v1 in eki) and (ejk[1] == eki[0]) and (eki[1] == eij[0]):
                            print(f'eij={eij}, ejk={ejk}, eki={eki}')
                            cons5.append(f'{formatVar(eij,"y")} + {formatVar(ejk,"y")} + {formatVar(eki,"y")} <= 2')
    return cons3, cons4, cons5
    
    
def writeATSPxy(WELfilename, LPfilename):
    G = nx.read_weighted_edgelist(WELfilename, create_using=nx.DiGraph)
    v1 = min(G.nodes)
    vars = createVariables(G)
    
    obj = prepareObjective(vars)
    cons1, cons2 = prepareAPconstraints(G,vars)
    cons3, cons4, cons5 = prepareSTEconstraints(v1, vars)
    
    with open(LPfilename, 'w') as lpf:
        lpf.write('\\\\ -------------------------------------\n')
        lpf.write('\\\\ This file was generated automaticaly. Do not modify it manually.\n')
        lpf.write(f'\\\\ Model {WELfilename.strip().split(",")[-1]}\n')
        lpf.write('\\\\ LP format - for model browsing. Use MPS format to Capture full model detail.\n')
        lpf.write('\\\\ -------------------------------------\n')
        
        lpf.write('\n\nMinimize\n')
        lpf.write(f'{obj}\n')
        
        lpf.write('\n\nSubject To\n')
        for c in cons1:
            lpf.write(f'{c}\n')
        lpf.write('\n')
        for c in cons2:
            lpf.write(f'{c}\n')
        lpf.write('\n')
        for c in cons3:
            lpf.write(f'{c}\n')
        lpf.write('\n')
        for c in cons4:
            lpf.write(f'{c}\n')
        lpf.write('\n')
        for c in cons5:
            lpf.write(f'{c}\n')
            
        lpf.write('\n\nBounds\nBinaries\n')
        var_str = ' '.join([f'{formatVar(e)}' for e in vars])
        lpf.write(var_str)
        
        lpf.write('\n\nEnd\n')


if __name__ == '__main__':
	ifname = ''
	ofname = ''
	try:
		for arg in sys.argv:
			if '=' in arg:
				parts = arg.split('=')
				if parts[0] == '--input' or parts[0] == '-i':
					ifname = parts[1]
				if parts[0] == '--output' or parts[0] == '-o':
					ofname = parts[1]
		assert ifname and ofname, 'SYNTAX: python convertWEL2LP.py -i=<input path/filename> -o=<output path/filename>'
	except AssertionError as msg:
		print(msg)

	try:
		writeATSPxy(ifname, ofname)
	except FileNotFoundError as msg:
		print(msg)
	

