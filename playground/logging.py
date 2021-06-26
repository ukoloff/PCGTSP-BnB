#
# Открыть файлы журналов
#
import sys
from pathlib import Path
from time import strftime
from contextlib import redirect_stdout

def start(name: str):
  folder = Path(__file__).parent.parent / 'logs'
  folder.mkdir(parents=True, exist_ok=True)
  moment = name + "." + strftime("%Y-%m-%d-%H-%M-%S")
  out = open(folder / (moment + '.log.txt'), 'w')
  nodes = open(folder / (moment + '.nodes.log.txt'), 'w')
  print("Redirecting to:", folder / moment, '...', file=sys.stderr)
  # redirect_stdout(out)
  sys.stdout = out
  return nodes
