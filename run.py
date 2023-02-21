import taichi as ti
from utils import *
from init import *
from gui import *
from sdf import *
from adaptive_gimp import AdaptiveGIMP
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=0)
parser.add_argument('--arch', default='cuda')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

grid_initializer = [None, initialize_mask1, initialize_mask2, initialize_mask3, initialize_mask4, initialize_mask5]
simulator = AdaptiveGIMP(dim = 2, 
                         level = 4,
                         sdf = SDF(dim=2),
                         coarsest_size = 16, 
                         n_particles = 20000, 
                         particle_initializer = lambda simulator: initialize_particle(simulator, args.case), 
                         grid_initializer = grid_initializer[args.case])
gui = GUI()

while True:
  for i in range(10): simulator.substep(2e-5)
  gui.show(simulator, True, True, True)