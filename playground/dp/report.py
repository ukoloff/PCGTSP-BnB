#
# Подбиваем итоги по запуску DP + Gurobi
#
import re
from pathlib import Path

logs = Path(__file__).parent.parent.parent / "logs/dp"

print("run\t\ttime\tstates\tsigmas\tLB\tLBs")
for log in logs.glob('*.log.txt'):
  time = None
  states, sigmas = 0, 0
  LBs = []
  for line in log.open():
    if m := re.match(r"Elapsed\s+time:\s+(\d+([.]\d*)?)", line):
      time = float(m[1])
    if m := re.match("layer\s+\d+\s+of\ssize\s+(\d+)\s+\((\d+)\)", line):
      states += int(m[1])
      sigmas += int(m[2])
    if m := re.search(r"Best\s+LB\s+is\s+(\d+([.]\d+)?)", line):
      LBs.append(float(m[1]))
  print(log.stem, time, states, sigmas, max(LBs, default=None), LBs, sep='\t')
