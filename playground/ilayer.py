#
# Информация об обработанном слое
#
from timeit import default_timer as timer

class iLayer:
  def __init__(self, size):
    self.start = timer()
    self.size = size
    self.nodes = 0
    self.skipped_nodes = 0
    self.LBs = []

  def found(self):
    self.nodes += 1

  def skipped(self):
    self.skipped_nodes += 1

  def dump(self, LB):
    print(f"Layer #{self.size} processed, {timer() - self.start:.1f} seconds")
    print(f"Nodes: {self.nodes}\tSkipped: {self.skipped_nodes}\tRatio: {self.nodes / (self.skipped_nodes + self.nodes) * 100:.0f}%\tLB: {LB}")
    print(flush=True)
