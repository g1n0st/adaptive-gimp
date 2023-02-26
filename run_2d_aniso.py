import taichi as ti
from utils import *
from init import *
from gui import *
from sdf import *
from adaptive_gimp import AdaptiveGIMP
import argparse
import time
import numpy as np

ti.init(arch = ti.gpu)

dt = 4.0e-5

# Material Parameters
E = 5000 # stretch

'''
gamma = 500 # shear
k = 1000 # normal
'''
gamma, k = 1e-5, 1e-5

# number of lines
# line space distance
dlx = 0.009
# type2 particle count per line

start_pos = ti.Vector([0.2, 0.5])

n_type2 = 200
n_type3 = n_type2 - 1

#line length
Length = 0.6
sl = Length/ (n_type2-1)

N23 = n_type2+n_type3
#type3
F = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # deformation gradient
D_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
d = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)

ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])

#get type2 from type3
@ti.func
def GetType2FromType3(index):
    return index, index+1   

@ti.kernel
def initialize(simulator : ti.template()):
    for i in range(n_type2):
        sq = i // n_type2
        simulator.x_p[i] = ti.Vector([start_pos[0]+ (i- sq* n_type2) * sl, start_pos[1] + sq* dlx])
        simulator.v_p[i] = ti.Matrix([0, 0])
        simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
        simulator.m_p[i] = simulator.p_mass
        simulator.g_p[i] = -1
        simulator.c_p[i] = ti.Vector([0.23, 0.33, 0.93])

    for i in range(n_type3):
        l, n = GetType2FromType3(i)
        simulator.x_p[i+n_type2] = 0.5*(simulator.x_p[l] + simulator.x_p[n])
        simulator.v_p[i+n_type2] = ti.Matrix([0, 0])
        simulator.F_p[i+n_type2] = ti.Matrix.identity(ti.f32, 2)
        simulator.m_p[i+n_type2] = simulator.p_mass
        simulator.g_p[i+n_type2] = -1
        

        F[i] = ti.Matrix([[1.0, 0.0],[0.0, 1.0]])
        dp0 = simulator.x_p[n] - simulator.x_p[l]
        dp1 = ROT90@dp0
        dp1 /= dp1.norm(1e-6)
        d[i] = ti.Matrix.cols([dp0,dp1])
        D_inv[i] = d[i].inverse() 

@ti.kernel
def get_force(solver : ti.template()):
    for p in range(N23):
      solver.f_p[p].fill(0.0)
    for p in range(n_type3):
        l, n = GetType2FromType3(p)
        Q, R = QR2(F[p])
        r11 = R[0,0]
        r12 = R[0,1]
        r22 = R[1,1] 
        A = ti.Matrix.rows([[E*r11*(r11-1)+gamma*r12**2, gamma * r12 * r22], \
                        [gamma * r12 * r22,  -k * (1 - r22)**2 * r22 * float(r22 <= 1)]])
        dphi_dF = Q @ A @ R.inverse().transpose() # Q.inverse().transpose() = Q.transpose().transpose() = Q
        Dp_inv_c0 = ti.Vector([D_inv[p][0,0],D_inv[p][1,0]])
        solver.f_p[l] += dphi_dF @ Dp_inv_c0
        solver.f_p[n] -= dphi_dF @ Dp_inv_c0

simulator = AdaptiveGIMP(dim = 2, 
                         level = 1,
                         sdf = HandlerSDF(2, np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float32), sphere_radius = 0.01),
                         lag_force = get_force,
                         coarsest_size = 256, 
                         n_particles = N23, 
                         particle_initializer = initialize, 
                         grid_initializer = initialize_mask1)


@ti.kernel
def update_particle_state(solver : ti.template()):
    for p in range(n_type3):
        l, n = GetType2FromType3(p)
        solver.v_p[p+n_type2] = 0.5 * (solver.v_p[l] + solver.v_p[n])
        solver.x_p[p+n_type2] = 0.5 * (solver.x_p[l] + solver.x_p[n])

        dp1 = solver.x_p[n] - solver.x_p[l]
        dp2 = ti.Vector([d[p][0,1],d[p][1,1]])
        # dp2 += dt * C3[p]@dp2
        d[p] = ti.Matrix.cols([dp1,dp2])
        F[p] = d[p] @ D_inv[p]


cf = 0.05
@ti.kernel
def return_mapping():
    for p in range(n_type3):
        Q,R = QR2(F[p])
        r12 = R[0,1]
        r22 = R[1,1]

        #cf = 0
        if r22 < 0:
            r12 = 0
            r22 = max(r22, -1)
        elif r22> 1:
            r12 = 0
            r22 = 1
        else:
            rr = r12**2
            zz = cf*(1.0 - r22)**2
            gamma_over_s = gamma/k
            f = gamma_over_s**2 * rr - zz**2
            if f > 0:
                scale = zz / ( gamma_over_s*  rr**0.5 )
                r12*= scale

        R[0,1] = r12
        R[1,1] = r22

        F[p] = Q@R
        d[p] = F[p]@D_inv[p].inverse()


def main():
    gui = ti.GUI("Cloth2D", (1024, 1024))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for _ in range(50):
            simulator.substep(dt)
            update_particle_state(simulator)
            return_mapping()

        gui.clear(0x112F41)

        x2_ny = simulator.x_p.to_numpy()[:n_type2, :]
        gui.circles(x2_ny[0 : n_type2], radius=2, color= 8123377)
        gui.show()

if __name__ == "__main__":
    main()
