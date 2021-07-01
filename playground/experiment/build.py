import re
import sys
from pathlib import Path

root = Path(__file__)
for i in range(3):
  root = root.parent

sources = []
for f in (root / "Salman/input").glob('*'):
    for line in open(f):
      if m := re.match(r"\s*GTSP_SETS\s*:\s*(\d+)", line):
        sources.append((int(m[1]), f.stem))

sources = sorted(sources)
print(*sources)
