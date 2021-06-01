import numpy as np
import networkx as nx

from itertools import combinations


def getInstance(filename):
	G = nx.read_weighted_edgelist(filename)
	clusters = {node: node for node in G.nodes}
	min_node = min(G.nodes)
	tree_edges = [(min_node, node) for node in G.nodes if node != min_node]

	tree = nx.DiGraph()
	tree.add_edges_from(tree_edges) 

	return G, clusters, tree


if __name__ == '__main__':
	G, clusters, tree = getInstance('../prefixes/1734.wel')
	print(f'Graph G: nodes\n{list(G.nodes)}')
	print(f'Clusters\n{list(clusters.keys())}')
	print(f'Tree\n{list(tree.edges)}')

