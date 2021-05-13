import sys
from pathlib import Path

root = Path(__file__).parent.parent
sroot = str(root)
sys.path.append(sroot)
import dp.fromPCGLNS as pcglns
sys.path.remove(sroot)

lib = root / "pcglns"

def load(name):
  return pcglns.getInstance(lib / f"{name}.pcglns")

def names():
  return (f.stem for f in lib.glob("*"))
