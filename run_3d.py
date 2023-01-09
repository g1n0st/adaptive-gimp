import taichi as ti
from utils import *
from gui_3d import *
from adaptive_gimp import AdaptiveGIMP
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

ti.init(arch=ti.cpu)

@ti.kernel
def init_mask(simulator : ti.template()):
  sz = simulator.finest_size
  for I in ti.grouped(ti.ndrange(sz, sz, sz)):
    simulator.activate_cell(simulator.level-1, I)

@ti.kernel
def init_p(simulator : ti.template()):
  for i in range(simulator.n_particles):
    simulator.x_p[i] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.3, ti.random() * 0.3 + 0.2]
    simulator.v_p[i] = [0, -20.0, 0]
    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
    simulator.m_p[i] = simulator.p_mass

simulator = AdaptiveGIMP(3, 2, 32, 50000, init_mask, init_p)
gui = GUI3D()

while True:
  for i in range(1): simulator.substep(1e-4)
  gui.show(simulator)