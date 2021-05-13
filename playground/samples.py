import sys
from pathlib import Path

root = Path(__file__).parent.parent
sroot = str(root)
sys.path.append(sroot)
import dp.Instance_generator as gen
import dp.fromPCGLNS as pcglns
sys.path.remove(sroot)

lib = root / "pcglns"


def load(name):
    return pcglns.getInstance(lib / f"{name}.pcglns")


def names():
    return (f.stem for f in lib.glob("*"))


def random(n, m):
    clusters = gen.clustering(n, m)
    tree = gen.tree_gen(m)
    order = gen.complete_order(tree)
    tour = gen.create_opt_tour(clusters, order)
    graph = gen.update_graph(gen.graph_generator(n), tour)
    return graph, clusters, tree
