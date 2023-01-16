import taichi as ti
from utils import *
from init import *
from gui import *
from adaptive_gimp import AdaptiveGIMP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=0)
args = parser.parse_args()

ti.init(arch=ti.cpu)

initializer = [initialize_mask0, initialize_mask1, initialize_mask2, initialize_mask3, initialize_mask4]
# simulator = AdaptiveGIMP(2, 1, 128, 20000, initializer[args.case], initialize_particle)
simulator = AdaptiveGIMP(2, 4, 16, 20000, initializer[args.case], initialize_particle)
gui = GUI()

while True:
  for i in range(20): simulator.substep(2e-5)
  gui.show(simulator)