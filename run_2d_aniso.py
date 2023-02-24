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

sdf = HandlerSDF(2, np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float32), sphere_radius = 0.01)

dim = 2
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
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

#type2
x = ti.Vector.field(2, dtype=float, shape=N23) # position 
v = ti.Vector.field(2, dtype=float, shape=N23) # velocity
f = ti.Vector.field(2, dtype=float, shape=N23) # lag force
C = ti.Matrix.field(2, 2, dtype=float, shape=N23) # affine velocity field

#type3
F = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # deformation gradient
D_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
d = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)

grid_v = ti.Vector.field(2, dtype= float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))

ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])

#get type2 from type3
@ti.func
def GetType2FromType3(index):
    return index, index+1   

@ti.kernel
def initialize():
    for i in range(n_type2):
        sq = i // n_type2
        x[i] = ti.Vector([start_pos[0]+ (i- sq* n_type2) * sl, start_pos[1] + sq* dlx])
        v[i] = ti.Matrix([0, 0])
        C[i] =  ti.Matrix([[0,0],[0,0]])

    for i in range(n_type3):
        l, n = GetType2FromType3(i)
        x[i+n_type2] = 0.5*(x[l] + x[n])
        v[i+n_type2] = ti.Matrix([0, 0])
        C[i+n_type3] =  ti.Matrix([[0,0],[0,0]])

        F[i] = ti.Matrix([[1.0, 0.0],[0.0, 1.0]])
        dp0 = x[n] - x[l]
        dp1 = ROT90@dp0
        dp1 /= dp1.norm(1e-6)
        d[i] = ti.Matrix.cols([dp0,dp1])
        D_inv[i] = d[i].inverse()  

'''
@ti.kernel
def init_p(simulator : ti.template()):
  for i in range(simulator.n_particles):
    xy = ti.Vector([0.2 + i / n_type2 * 0.6, 0.5])
    simulator.x_p[i] = xy
    simulator.v_p[i] = [0.0, 0.0]

    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
    simulator.m_p[i] = simulator.p_mass

    simulator.g_p[i] = -1
    simulator.c_p[i] = ti.Vector([0.23, 0.33, 0.93])

simulator = AdaptiveGIMP(dim = 2, 
                         level = 1,
                         sdf = sdf,
                         lag_force = None,
                         coarsest_size = 32, 
                         n_particles = n_type2, 
                         particle_initializer = init_p, 
                         grid_initializer = initialize_mask1)
'''

@ti.kernel
def Get_Force():
    for p in range(N23):
      f[p].fill(0.0)
    for p in range(n_type3):
        l, n = GetType2FromType3(p)
        Q, R = QR2(F[p])
        r11 = R[0,0]
        r12 = R[0,1]
        r22 = R[1,1] 
        A = ti.Matrix.rows([[E*r11*(r11-1)+gamma*r12**2, gamma * r12 * r22], \
                        [gamma * r12 * r22,  -k * (1 - r22)**2 * r22 * float(r22 <= 1)]])
        dphi_dF = Q @ A @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q
        Dp_inv_c0 = ti.Vector([D_inv[p][0,0],D_inv[p][1,0]])
        f[l] += dphi_dF @ Dp_inv_c0
        f[n] -= dphi_dF @ Dp_inv_c0

@ti.kernel
def Particle_To_Grid():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C[p]
        mass = 1.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * (mass * (v[p]+affine@dpos) + f[p] * dt)
    
bound = 3
@ti.kernel
def Grid_Collision():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.80

            #circle collision
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0
            
            
            pos = (ti.Vector([i, j]) + 0.5) * dx
            fixed, inside, dotnv, diff_vel, n = sdf.check(pos, grid_v[i, j])
            if inside:
              if fixed: grid_v[i, j].fill(0.0)


@ti.kernel
def Grid_To_Particle():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[p] = new_v
        x[p] += dt * v[p]
        # C2[p] = new_C


@ti.kernel
def Update_Particle_State():
    for p in range(n_type3):
        l, n = GetType2FromType3(p)
        v[p+n_type2] = 0.5 * (v[l] + v[n])
        x[p+n_type2] = 0.5 * (x[l] + x[n])

        dp1 = x[n] - x[l]
        dp2 = ti.Vector([d[p][0,1],d[p][1,1]])
        # dp2 += dt * C3[p]@dp2
        d[p] = ti.Matrix.cols([dp1,dp2])
        F[p] = d[p] @ D_inv[p]


cf = 0.05
@ti.kernel
def Return_Mapping():
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



@ti.kernel
def Reset():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

def main():
    initialize()

    gui = ti.GUI("Cloth2D", (1024, 1024))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for _ in range(50):
            Reset()
            Get_Force()
            Particle_To_Grid()
            Grid_Collision()
            Grid_To_Particle()
            Update_Particle_State()
            Return_Mapping()

        gui.clear(0x112F41)

        x2_ny = x.to_numpy()[:n_type2, :]
        gui.circles(x2_ny[0 : n_type2], radius=2, color= 8123377)
        gui.show()

if __name__ == "__main__":
    main()
