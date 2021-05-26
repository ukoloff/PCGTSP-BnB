import networkx as nx

from klasses import Task, STNode
import nc, samples, nc0

task = samples.random(27, 7)
nc0.nc0(task)
x = STNode(task, (1, 5, 7, 3, 6))
nc.lower_bounds(x)
print(x.bounds)
