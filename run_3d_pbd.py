import taichi as ti
from utils import *
from init import *
from gui_3d import *
from sdf import *
from adaptive_gimp import AdaptiveGIMP
import argparse
import time
import numpy as np

ti.init(arch = ti.gpu)

dt = 1e-4

XPBD_iter = 20
stretch_compliance = 1e-7
stretch_relaxation = 0.2

start_pos = ti.Vector([0.3, 0.3])

L = 100
NPLANE = np.array([L] * 2)
NV = L**2
NE = (L-1)*L*2+(L-1)**2
dx = 1 / (L - 1)

E_verts = ti.Vector.field(2, dtype=ti.i32, shape=NE)
la = ti.field(ti.f32, shape=NE)

ox = ti.Vector.field(3, ti.f32, shape=NV)
invM = ti.field(ti.f32, shape=NV)
new_x = ti.Vector.field(3, ti.f32, shape=NV)
dp = ti.Vector.field(3, ti.f32, shape=NV)


@ti.func
def i2p(I):
    return I.x * NPLANE[1] + I.y

@ti.func
def set_edge(e, I0, I1):
    E_verts[e] = ti.Vector([i2p(I0), i2p(I1)])

@ti.kernel
def get_vertices():
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
        if I.x < L-1: set_edge(I.x*L+I.y, I, I+[1, 0])
        if I.y < L-1: set_edge((L-1)*L+I.y*L+I.x, I, I+[0, 1])
        if I.x < L-1 and I.y < L-1: set_edge((L-1)*L*2+I.y*(L-1)+I.x, I, I+[1, 1])

get_vertices()

@ti.kernel
def initialize(simulator : ti.template()):
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
        simulator.x_p[i2p(I)] = ti.Vector([start_pos[0] + I[0] * dx * 0.4, 0.5, start_pos[1] + I[1] * dx * 0.4])

    for i in range(NV):
        simulator.m_p[i] = simulator.p_mass
        invM[i] = 1. / simulator.p_mass
        ox[i] = simulator.x_p[i]

        simulator.v_p[i] = ti.Vector([0, 0, 0])
        simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
        simulator.g_p[i] = 0
        simulator.c_p[i] = ti.Vector([0.23, 0.33, 0.93])

@ti.kernel
def apply_ext_force(solver : ti.template()):
    for p in range(NV):
        new_x[p] = solver.x_p[p] + solver.v_p[p] * dt

@ti.kernel
def solve_stretch(solver : ti.template()):
    for v in range(NV):
        dp[v].fill(0.0)

    for e in range(NE):
        v0, v1 = E_verts[e][0], E_verts[e][1]
        w1, w2 = invM[v0], invM[v1]
        if w1 + w2 > 0.:
            n = new_x[v0] - new_x[v1]
            d = n.norm()
            rest_len = (ox[v0] - ox[v1]).norm()
            _dp = ti.zero(n)
            constraint = (d - rest_len)
            compliance = stretch_compliance / (dt**2)
            d_lambda = -(constraint + compliance * la[e]) / (w1 + w2 + compliance) * stretch_relaxation
            _dp = d_lambda * n.normalized(1e-12) # eq. (17)
            la[e] += d_lambda

            dp[v0] += _dp * w1
            dp[v1] -= _dp * w2
    
    for v in range(NV):
        new_x[v] += dp[v]

@ti.kernel
def get_pbd_force(solver : ti.template()):
    for v in range(NV):
        new_v = (new_x[v] - solver.x_p[v]) / dt
        solver.f_p[v] = (new_v - solver.v_p[v]) / dt

def get_force(solver):
    apply_ext_force(solver)
    la.fill(0.0)
    for iter in range(XPBD_iter):
        solve_stretch(solver)
    
    get_pbd_force(solver)

@ti.func
def init_grid(simulator, I):
  sz = simulator.finest_size
  L = -1
  if I[0] < sz / 2:
    L = 0
    simulator.activate_cell(0, I)
  else: 
    L = 1
    simulator.activate_cell(1, I)
  return L

@ti.kernel
def visualize(simulator : ti.template()):
    for p in range(NV):
        if simulator.x_p[p][0] < 0.5:
            simulator.c_p[p] = ti.Vector([0.23, 0.33, 0.93])
        else:
            simulator.c_p[p] = ti.Vector([0.93, 0.33, 0.23])

# sdf = SphereSDF(dim=3, radius = 0.1, pos = ti.Vector([0.5, 0.15, 0.5]))
sdf = HandlerSDF(dim=3, pos=np.array([[0.3, 0.5, 0.3], [0.3, 0.5, 0.7], [0.7, 0.5, 0.3], [0.7, 0.5, 0.7]], dtype=np.float32), sphere_radius = 0.1)

simulator = AdaptiveGIMP(dim = 3, 
                         level = 2,
                         sdf = sdf,
                         lag_force = get_force,
                         coarsest_size = 128, 
                         n_particles = NV, 
                         particle_initializer = initialize, 
                         grid_initializer = init_grid)

def main():
    gui = GUI3D(simulator, res=2048)
    while True:
        for _ in range(10):
            simulator.substep(dt)
        visualize(simulator)
        gui.show()

if __name__ == "__main__":
    main()
