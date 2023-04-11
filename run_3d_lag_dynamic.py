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
E, nu = 5e5, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1e4

start_pos = ti.Vector([0.3, 0.3, 0.4])

L = 30
NCUBE = np.array([L] * 3)
NV = L**3
NC = 5 * np.product(NCUBE - 1)
dx = 1 / (NCUBE.max() - 1)

F_verts = ti.Vector.field(4, dtype=ti.i32, shape=NC)
F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NC)
F_W = ti.field(dtype=ti.f32, shape=NC)

@ti.func
def i2p(I):
    return (I.x * NCUBE[1] + I.y) * NCUBE[2] + I.z

@ti.func
def set_element(e, I, verts):
    for i in ti.static(range(3 + 1)):
        F_verts[e][i] = i2p(I + (([verts[i] >> k for k in range(3)] ^ I) & 1))


@ti.kernel
def get_vertices():
    '''
    This kernel partitions the cube into tetrahedrons.
    Each unit cube is divided into 5 tetrahedrons.
    '''
    for I in ti.grouped(ti.ndrange(*(NCUBE - 1))):
        e = ((I.x * (NCUBE[1] - 1) + I.y) * (NCUBE[2] - 1) + I.z) * 5
        for i, j in ti.static(enumerate([0, 3, 5, 6])):
            set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
        set_element(e + 4, I, (1, 2, 4, 7))

get_vertices()

@ti.func
def Ds(verts, solver):
    return ti.Matrix.cols([solver.x_p[verts[i]] - solver.x_p[verts[3]] for i in range(3)])

@ti.kernel
def initialize(simulator : ti.template()):
    for I in ti.grouped(ti.ndrange(*(NCUBE))):
        simulator.x_p[i2p(I)] = (start_pos + I * dx * 0.4)
        if any(I < 3) or any(I > L - 3):
          simulator.g_p[i2p(I)] = 1
        else:
          simulator.g_p[i2p(I)] = 0
    
    for c in range(NC):
        F = Ds(F_verts[c], simulator)
        F_B[c] = F.inverse()
        F_W[c] = ti.abs(F.determinant()) / 6
        for i in range(4):
            simulator.m_p[F_verts[c][i]] += F_W[c] / 4 * density

    for i in range(NV):
        simulator.v_p[i] = ti.Vector([0, 0, 0])
        simulator.F_p[i] = ti.Matrix.identity(ti.f32, 3)
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
    for c in range(NC):
      F = Ds(F_verts[c], solver) @ F_B[c]
      P = PK1(mu, la, F)
      H = -F_W[c] * P @ F_B[c].transpose()
      for ii in ti.static(range(3)):
          fi = ti.Vector([H[0, ii], H[1, ii], H[2, ii]])
          solver.f_p[F_verts[c][ii]] += fi
          solver.f_p[F_verts[c][3] ] += -fi

@ti.kernel
def visualize(simulator : ti.template()):
    for p in range(NV):
        if simulator.g_p[p] == 0:
            simulator.c_p[p] = ti.Vector([0.23, 0.33, 0.93])
        else:
            simulator.c_p[p] = ti.Vector([0.93, 0.33, 0.23])

simulator = AdaptiveGIMP(dim = 3, 
                         level = 2,
                         sdf = SphereSDF(dim=3, radius = 0.1, pos = ti.Vector([0.5, 0.15, 0.5])),
                         lag_force = get_force,
                         coarsest_size = 64, 
                         n_particles = NV, 
                         particle_initializer = initialize, 
                         grid_initializer = None)

def main():
    gui = GUI3D(simulator, res=2048)
    while True:
        for _ in range(10):
            simulator.substep(dt)
        visualize(simulator)
        gui.show()

if __name__ == "__main__":
    main()
