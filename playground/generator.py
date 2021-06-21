#
# Массовая генерация суффиксных графов
# Для тестирования интерфейса к Gurobi
#
import sys
from klasses import Task, STNode
import samples
import nc
import children

src = "e5x_1" if len(sys.argv) < 2 else sys.argv[1]
print("Loading:", src)

# task = samples.random(1000, 12)
task = samples.load(src)

print("Size:\t", task.dists.order(), '/', len(task.clusters))
print("UB:\t", task.UB)
# print("Solution so far:\t", task.solution)
print()

nc.initL1(task)
nc.initL2(task)

root = STNode(task)
for node in children.subtree(root):
    print(node.sigma, end='\t', flush=True)

    graph = nc.nc(node, L=1)
    tree = nc.get_order(node, graph)
    # Use them!
    print(len(graph), len(tree))
