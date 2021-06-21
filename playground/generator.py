#
# Массовая генерация суффиксных графов
# Для тестирования интерфейса к Gurobi
#
import sys
from klasses import Task, STNode
import samples
import nc
import children

# src = "e5x_1" if len(sys.argv) < 2 else sys.argv[1]
src = "e5x_1"
print("Loading:", src)

# task = samples.random(1000, 12)
task = samples.load(src)


def make_generator(task: Task):
    print("Size:\t", task.dists.order(), '/', len(task.clusters))
    print("UB:\t", task.UB)
    # print("Solution so far:\t", task.solution)
    print()

    nc.initL1(task)
    nc.initL2(task)

    root = STNode(task)
    for node in children.subtree(root):
        # print(node.sigma, end='\t', flush=True)

        graph = nc.nc(node, L=1)
        tree = nc.get_order(node, graph)

        yield graph, tree, node.sigma


it = make_generator(task)


def next_graph():
    """Возвращает следуюшую тройку:
    1) Граф кластеров
    2) Дерево (DAG) порядка
    3) Сигма
    """
    return next(it)


# Пример использования:
for i in range(5):
    graph, tree, sigma = next_graph()
    print(sigma, len(graph), len(tree))
