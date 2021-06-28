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


class STNode:
    """Узел дерева поиска
    """
    def __init__(self, task, sigma=(1,)):
        self.task = task
        self.sigma = sigma
        self.parent = None
        self.allowed_groups = None

    def S(self):
      return (frozenset(self.sigma), self.sigma[-1])

    def is_leaf(self):
      return len(self.sigma) == len(self.task.clusters)

    def gap(self):
      LB = self.LB
      UB = self.task.UB
      return 0 if LB == 0 else (UB - LB) / LB * 100
