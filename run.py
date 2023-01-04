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

initializer = [initialize_mask1, initialize_mask2, initialize_mask3, initialize_mask4]
simulator = AdaptiveGIMP(4, initializer[args.case], initialize_particle)
gui = GUI()

while True:
  for i in range(20): simulator.substep(dt)
  gui.show(simulator)