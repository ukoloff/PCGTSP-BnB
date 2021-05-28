import sys
import os
from pathlib import Path

import networkx as nx
import numpy as np

from Instance_generator import get_instance
EDGE_WEIGHT_SECTION = 'EDGE_WEIGHT_SECTION'
GTSP_SET_SECTION = 'GTSP_SET_SECTION'
GTSP_SET_ORDERING = 'GTSP_SET_ORDERING'

VERBOSE = True

def make_header(problem_name, n, m):
    header = f'NAME : {problem_name}\nTYPE : PCGLNS\nCOMMENT : no comments\nDIMENSION : {n}\nGTSP_SETS : {m}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHTS_FORMAT : FULL_MATRIX\n'
    return header

def make_edge_weights(G):
    lines=[EDGE_WEIGHT_SECTION]
    n = len(G.nodes)
    
    for v in range(n):
        line = ' '.join([f'{G[v+1][u+1]["weight"]}' if u != v else '0' for u in range(n)])
        lines.append(line)
    return lines

def make_gtsp_sets(clusters):
    lines=[GTSP_SET_SECTION]
    m =len(clusters)
    
    for c_idx in range(1,m+1):
        line = f'{c_idx} ' + ' '.join([f'{v}' for v in clusters[c_idx]]) + ' -1'
        lines.append(line)
        
    return lines

def make_gtsp_sets_ordering(tree, m):
    lines=[GTSP_SET_ORDERING]
    for c in range(2,m+1):
        if c in tree:
            desc = nx.descendants(tree,c)
            if desc:
                line = f'{c} ' + ' '.join([f'{d}' for d in desc]) + ' -1\n'
                lines.append(line)
    return lines
            
def make_footer(first_cluster):
    return f'START_GROUP_SECTION\n{first_cluster}\nEND'
    
    
def main(fname, n, m):
    G, clusters, tree = get_instance(n,m)
    
    if VERBOSE:
        print('==============================')
        print(f'instance of {n} nodes and {m} clusters is ready')
        print(f'writing to {fname} is started')

    
    with open(fname, 'w') as fout:
        header = make_header(fname, n, m)
        fout.write(header)
        if VERBOSE:
            print('header is written')
        
        weights_section = make_edge_weights(G)
        fout.write('\n'.join(weights_section) + '\n')
        if VERBOSE:
            print(f'{EDGE_WEIGHT_SECTION} is ready')

        
        gtsp_section = make_gtsp_sets(clusters)
        fout.write('\n'.join(gtsp_section) + '\n')
        if VERBOSE:
            print(f'{GTSP_SET_SECTION} is ready')

        
        order_section = make_gtsp_sets_ordering(tree, m)
        fout.write('\n'.join(order_section) + '\n')
        if VERBOSE:
            print(f'{GTSP_SET_ORDERING} is ready')
        
        footer = make_footer(1)
        fout.write(footer)
        if VERBOSE:
            print(f'Footer is written')
            print('==============================')
        print(f'Random PCGLNS model {fname} is ready')

if __name__ == '__main__':

    fname = ''
    n = 0
    m = 0
    try:
        for arg in sys.argv:
            if arg == '-s':
                VERBOSE = False

            if '=' in arg:
                parts = arg.split('=')
                if parts[0] == '-o':
                    fname = parts[1]
                if parts[0] == '-m':
                    m = int(parts[1])
                if parts[0] == '-n':
                    n = int(parts[1])

        assert fname and n and m

    except:
        print('SYNTAX: python randomPCGLNS.py -o <output file name> -n <no. of nodes> -m <no. of clusters>')

    main(fname, n, m)
