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

# Material Parameters
E, nu = 5e6, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1e4

start_pos = ti.Vector([0.3, 0.3, 0.4])

L = 400
NPLANE = np.array([L] * 2)
NV = L**2
NF = ((L-1)**2)*2
dx = 1 / (L - 1)

F_verts = ti.Vector.field(3, dtype=ti.i32, shape=NF)
F_B = ti.Matrix.field(2, 3, dtype=ti.f32, shape=NF)
F_W = ti.field(dtype=ti.f32, shape=NF)

@ti.func
def i2p(I):
    return I.x * NPLANE[1] + I.y

@ti.func
def set_triangle(f, I0, I1, I2):
    F_verts[f] = ti.Vector([i2p(I0), i2p(I1), i2p(I2)])


@ti.kernel
def get_vertices():
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
      if all(I < L-1):
        set_triangle((I.x*(L-1)+I.y)*2, I, I+[1, 0], I+[0, 1])
        set_triangle((I.x*(L-1)+I.y)*2+1, I+[1, 1], I+[0, 1], I+[1, 0])

get_vertices()

@ti.func
def Ds(verts, solver):
    return ti.Matrix.cols([solver.x_p[verts[i]] - solver.x_p[verts[2]] for i in range(2)])

@ti.kernel
def initialize(simulator : ti.template()):
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
        simulator.x_p[i2p(I)] = ti.Vector([start_pos[0] + I[0] * dx * 0.4, 0.5, start_pos[1] + I[1] * dx * 0.4])
    
    for c in range(NF):
        F = Ds(F_verts[c], simulator)
        F_B[c] = (F.transpose() @ F).inverse() @ F.transpose()
        F_W[c] = 1.0
        for i in range(3):
            simulator.m_p[F_verts[c][i]] += F_W[c] / 3 * density

    for i in range(NV):
        simulator.v_p[i] = ti.Vector([0, 0, 0])
        simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
        simulator.g_p[i] = 0
        simulator.c_p[i] = ti.Vector([0.23, 0.33, 0.93])

@ti.func
def PK1(u, l, F):
    U, sig, V = ti.svd(F, ti.f32)
    R = U @ V.transpose()
    J = F.determinant()
    return 2 * u * (F - R) + l * (J - 1) * J * F.inverse().transpose()

@ti.kernel
def get_force(solver : ti.template()):
    for p in range(NV):
      solver.f_p[p].fill(0.0)
    for c in range(NF):
      F = Ds(F_verts[c], solver) @ F_B[c]
      P = PK1(mu, la, F)
      H = -F_W[c] * P @ F_B[c].transpose()
      for ii in ti.static(range(2)):
          fi = ti.Vector([H[0, ii], H[1, ii], H[2, ii]])
          solver.f_p[F_verts[c][ii]] += fi
          solver.f_p[F_verts[c][2]] += -fi

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

# sdf = SphereSDF(dim=3, radius = 0.07, pos = ti.Vector([0.5, 0.35, 0.5]))
sdf = HandlerSDF(dim=3, pos=np.array([[0.3, 0.5, 0.3], [0.3, 0.5, 0.7], [0.7, 0.5, 0.3], [0.7, 0.5, 0.7]], dtype=np.float32), sphere_radius = 0.1)

simulator = AdaptiveGIMP(dim = 3,
                         level = 2,
                         sdf = sdf,
                         lag_force = get_force,
                         coarsest_size = 128,
                         n_particles = NV,
                         particle_initializer = initialize,
                         grid_initializer = init_grid)

# exporter = MeshExporter("./results/cloth")

def main():
    frame = 0
    gui = GUI3D(simulator, res=2048)
    while True:
        for _ in range(10):
            simulator.substep(dt)
        visualize(simulator)
        gui.show()
        # if frame % 40 == 0: exporter.export(simulator.x_p.to_numpy(), F_verts.to_numpy().reshape(-1, 3), frame // 40)
        frame += 1

if __name__ == "__main__":
    main()
