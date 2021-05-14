import sys
from pathlib import Path

import klasses

root = Path(__file__).parent.parent
sroot = str(root)
sys.path.append(sroot)
import dp.Instance_generator as gen
import dp.fromPCGLNS as pcglns
sys.path.remove(sroot)

lib = root / "pcglns"


def load(name):
    """(str) -> Task
    """
    return klasses.Task(*pcglns.getInstance(lib / f"{name}.pcglns"))


def names():
    return (f.stem for f in lib.glob("*"))


def random(n, m):
    """(int, int) -> Task
    """
    clusters = gen.clustering(n, m)
    tree = gen.tree_gen(m)
    order = gen.complete_order(tree)
    tour = gen.create_opt_tour(clusters, order)
    graph = gen.update_graph(gen.graph_generator(n), tour)
    return klasses.Task(graph, clusters, tree)
