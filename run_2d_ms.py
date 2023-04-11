import taichi as ti
from utils import *
from init import *
from gui import *
from sdf import *
from adaptive_gimp import AdaptiveGIMP
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='cuda')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

DT = 5e-5
NP = 128
stiffness1 = 1e7
stiffness2 = 1e7
eps = 1e-6

x0_p = ti.Vector.field(2, float, shape=NP)

@ti.kernel
def init_p(simulator : ti.template()):
  for i in range(simulator.n_particles):
    xy = ti.Vector([0.2 + i / NP * 0.6, 0.5])
    simulator.x_p[i] = xy
    x0_p[i] = xy
    simulator.v_p[i] = [0.0, -10.0]

    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
    simulator.m_p[i] = simulator.p_mass * 0.1

    simulator.g_p[i] = -1
    simulator.c_p[i] = ti.Vector([0.23, 0.33, 0.93])

sdf = HandlerSDF(2, np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float32), sphere_radius = 0.05)

@ti.kernel
def mass_spring(solver : ti.template()):
  for i in range(NP):
      solver.f_p[i].fill(0.0)
  # stretch
  for i in range(solver.n_particles - 1):
    disp = solver.x_p[i] - solver.x_p[i+1]
    rest_disp = x0_p[i] - x0_p[i+1]
    spring_force = -stiffness1 * (disp.norm() - rest_disp.norm()) ** 2 * disp.normalized()
    solver.f_p[i] += spring_force
    solver.f_p[i+1] -= spring_force

  # bending
  for i in range(solver.n_particles - 2):
    disp = solver.x_p[i] - solver.x_p[i+2]
    rest_disp = x0_p[i] - x0_p[i+2]
    spring_force = -stiffness2 * (disp.norm() - rest_disp.norm()) ** 2 * disp.normalized()
    solver.f_p[i] += spring_force
    solver.f_p[i+2] -= spring_force

@ti.func
def _2v2(simulator, I):
  sz = simulator.finest_size
  L = -1
  if I[0] < sz / 2:
    L = 0
    simulator.activate_cell(0, I)
  else: 
    L = 1
    simulator.activate_cell(1, I)
  return L


simulator = AdaptiveGIMP(dim = 2, 
                         level = 2,
                         sdf = sdf,
                         lag_force = mass_spring,
                         coarsest_size = 64, 
                         n_particles = NP, 
                         particle_initializer = init_p, 
                         grid_initializer = _2v2)
gui = GUI()

while True:
  for i in range(20): simulator.substep(DT)
  gui.show(simulator, True, True, True)