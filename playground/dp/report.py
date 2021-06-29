#
# Подбиваем итоги по запуску DP + Gurobi
#
import re
from pathlib import Path

logs = Path(__file__).parent.parent.parent / "logs/dp"
for log in logs.iterdir():
  time = None
  for line in log.open():
    if m := re.match("Elapsed\s+time:\s+(\d+([.]\d*))", line):
      time = float(m[1])
  print(log.stem, time, sep='\t')
