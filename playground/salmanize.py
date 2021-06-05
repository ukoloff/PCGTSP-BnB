#
# "Чистый" код для подсчёта пути по "внешнему" пути (суффиксу)
# NC + APa
import networkx as nx


def nc0(dists: nx.DiGraph, clusters: dict, order: nx.DiGraph):
    """Построить граф кластеров для данной задачи
    (с частичным учётом оганичений предшествования)
    """
