#
# Построение модели для Gurobi
# Альтернативная версия
#
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

def model(graph: nx.DiGraph):
  m = gp.Model('pctsp')
