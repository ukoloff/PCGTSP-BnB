#
# Data classes
#
import networkx as nx


class Task:
    """Задача PCGTSP
    """

    def __init__(self, dists, clusters, tree):
        self.dists = dists
        self.clusters = clusters
        self.tree = nx.transitive_reduction(tree)
        self.tree_closure = nx.transitive_closure_dag(tree)


class Branch:
    """Узел дерева
    """
    def __init__(self, task, sigma=(1,)):
        self.task = task
        self.sigma = sigma
        self.S = (set(sigma[:-1]), sigma[-1])
