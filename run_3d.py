import taichi as ti
from utils import *
from gui_3d import *
from init import *
from adaptive_gimp import AdaptiveGIMP
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

ti.init(arch=ti.cpu, kernel_profiler=True)

@ti.kernel
def init_p(simulator : ti.template()):
  for i in range(simulator.n_particles):
    if i // 25000 == 0:
      simulator.x_p[i] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.1, ti.random() * 0.3 + 0.2]
      simulator.v_p[i] = [0, 20.0, 0]
    else:
      simulator.x_p[i] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.5, ti.random() * 0.3 + 0.2]
      simulator.v_p[i] = [0, -20.0, 0]

    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
    simulator.m_p[i] = simulator.p_mass
    simulator.g_p[i] = simulator.level-1

simulator = AdaptiveGIMP(3, 2, 32, 50000, init_p, None)
gui = GUI3D()

frame = 0
while frame < 5000:
  for i in range(1): simulator.substep(1e-4)
  gui.show(simulator)
  frame += 1

ti.profiler.print_kernel_profiler_info()