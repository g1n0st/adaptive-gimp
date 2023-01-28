import taichi as ti
from utils import *
from gui_3d import *
from init import *
from adaptive_gimp import AdaptiveGIMP
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='cuda')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch), kernel_profiler=True)

@ti.kernel
def init_p(simulator : ti.template()):
  for i in range(simulator.n_particles):
    xyz = ti.Vector([ti.random(), ti.random(), ti.random()])
    if i // 25000 == 0:
      simulator.x_p[i] = [xyz[0] * 0.3 + 0.2, xyz[1] * 0.3 + 0.1, xyz[2] * 0.3 + 0.2]
      simulator.v_p[i] = [0, 20.0, 0]
    else:
      simulator.x_p[i] = [xyz[0] * 0.3 + 0.2, xyz[1] * 0.3 + 0.5, xyz[2] * 0.3 + 0.2]
      simulator.v_p[i] = [0, -20.0, 0]

    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
    simulator.m_p[i] = simulator.p_mass
    if all(0.05 <= xyz <= 0.95):
      simulator.g_p[i] = 0
    elif all(0.01 <= xyz <= 0.99):
      simulator.g_p[i] = 1
    else:
      simulator.g_p[i] = 2

simulator = AdaptiveGIMP(3, 3, 16, 50000, init_p, None)
gui = GUI3D(simulator)

frame = 0
while frame < 5000:
  for i in range(1): simulator.substep(1e-4)
  gui.show()
  frame += 1

ti.profiler.print_kernel_profiler_info()