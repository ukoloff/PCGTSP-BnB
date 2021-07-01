#!/usr/bin/env python

import re
import sys
from pathlib import Path

root = Path(__file__).resolve()
for i in range(3):
  root = root.parent

samples = root / "Salman"
tasks = samples / "input"
logs = root / "logs/s/bnb"
logs.mkdir(parents=True, exist_ok=True)

sources = []
for f in tasks.glob('*'):
    for line in open(f):
      m = re.match(r"\s*GTSP_SETS\s*:\s*(\d+)", line)
      if m:
        sources.append((int(m[1]), f.stem))

sources = sorted(sources)
# print(*sources)

def UB(sample):
  for line in (samples / "heuristic" / f"{sample}.pcglns.result.txt").open():
    m = re.match(r"\s*Cost\s*:\s*(\d+)", line)
    if m:
      return float(m[1])

def header():
  print("""
#!/bin/bash

export GRB_LICENSE_FILE=~/gurobi.lic
export LD_PRELOAD=~/ioctl/ioctl_faker.so

threads=36
m="--mem=128G"
g="--cpus-per-task=$threads"
t="--time=600"
p="--partition=apollo"
mode="-mode=default"
#max_time="-max_time=10800"

source ../../vi/mine/bin/activate
""".strip())

header()
for n, f in sources:
  thisUB = UB(f)
  print(f"""
# {f}[{n}] UB={thisUB}
outf="--output={logs}/{f}.log.txt"
errf="--error={logs}/{f}.errors.log.txt"
cmd="srun $m $t $g $p -u python ../walk.py s/{f} --gap 5"
sbatch $t $m $p $errf $outf $g -J .{f} --wrap="$cmd"
""")
