import re
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
solutions = root / "heuristic"


def load(name):
    """(str) -> Task
    """
    if name.startswith('s/'):
      name = name[2:]
      Salman = root / "Salman"
      src = Salman / "input"
      res = Salman / "heuristic"
    else:
      src = root / "pcglns"
      res = root / "heuristic"

    result = klasses.Task(*pcglns.getInstance(src / f"{name}.pcglns"))
    result.UB, order = read_solution(res / f"{name}.pcglns.result.txt")
    backidx = {i: n for n, points in result.clusters.items() for i in points}
    result.solution = [(backidx[i], i) for i in order]
    return result

def read_solution(name):
    UB = None
    order = None
    with open(name) as f:
      for line in f:
        line = line.split(":", 1)
        if len(line) < 2:
          continue
        k, v = line
        k = k.strip()
        if k == 'Cost':
          UB = float(v)
        if k == 'Tour Ordering':
          order = [int(i) for i in re.sub(r"\D+", " ", v).strip().split()]
    return UB, order

def names():
    return (f.stem for f in lib.glob("*"))


def names_salman():
    return (f.stem for f in (root / "Salman/input").glob('*'))

def random(n, m):
    """(int, int) -> Task
    """
    clusters = gen.clustering(n, m)
    tree = gen.tree_gen(m)
    order = gen.complete_order(tree)
    tour = gen.create_opt_tour(clusters, order)
    graph = gen.update_graph(gen.graph_generator(n), tour)
    result = klasses.Task(graph, clusters, tree)
    result.UB = 0
    result.solution = list(zip(order, tour))
    return result

if __name__ == '__main__':
  A = random(27, 7)
  print(A)
  B = load("e1x_1")
  print(B)
  C = load("s/br17.12")
  print(C)
