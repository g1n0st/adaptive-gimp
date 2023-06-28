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

indices = ti.field(ti.i32, shape=NF * 3)
F_verts = ti.Vector.field(3, dtype=ti.i32, shape=NF)
F_color = ti.Vector.field(3, dtype=ti.f32, shape=NF)
F_B = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NF)
F_W = ti.field(dtype=ti.f32, shape=NF)

@ti.func
def i2p(I):
    return I.x * NPLANE[1] + I.y

@ti.func
def set_triangle(f, I0, I1, I2):
    F_verts[f] = ti.Vector([i2p(I0), i2p(I1), i2p(I2)])
    indices[f*3+0] = i2p(I0)
    indices[f*3+1] = i2p(I1)
    indices[f*3+2] = i2p(I2)

@ti.kernel
def get_vertices():
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
      if all(I < L-1):
        set_triangle((I.x*(L-1)+I.y)*2, I, I+[1, 0], I+[0, 1])
        set_triangle((I.x*(L-1)+I.y)*2+1, I+[1, 1], I+[0, 1], I+[1, 0])

get_vertices()

@ti.kernel
def initialize(simulator : ti.template()):
    for I in ti.grouped(ti.ndrange(*(NPLANE))):
        simulator.x_p[i2p(I)] = ti.Vector([start_pos[0] + I[0] * dx * 0.4, 0.5, start_pos[1] + I[1] * dx * 0.4])
    
    for c in range(NF):
        D0 = simulator.x_p[F_verts[c][0]] - simulator.x_p[F_verts[c][2]]
        D1 = simulator.x_p[F_verts[c][1]] - simulator.x_p[F_verts[c][2]]
        D2 = D0.cross(D1).normalized()
        F = ti.Matrix.cols([D0, D1, D2])
        F_B[c] = F.inverse()
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
      d0 = simulator.x_p[F_verts[c][0]] - simulator.x_p[F_verts[c][2]]
      d1 = simulator.x_p[F_verts[c][1]] - simulator.x_p[F_verts[c][2]]
      d2 = d0.cross(d1).normalized()
      d = ti.Matrix.cols([d0, d1, d2])
      F = d @ F_B[c]
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
def visualize(solver : ti.template()):
  for v3 in range(NF):
    angle_m = ti.Vector([0.0, 0.0, 0.0])
    p0 = indices[v3*3+0]
    p1 = indices[v3*3+1]
    p2 = indices[v3*3+2]
    c1 = 3e2
    c3 = 3e1
    C_x = (solver.x_p[p0] + solver.x_p[p1] + solver.x_p[p2]) / 3.0
    angle_m += ((solver.x_p[p0] - C_x)*c1).cross(solver.v_p[p0]*c3)
    angle_m += ((solver.x_p[p1] - C_x)*c1).cross(solver.v_p[p1]*c3)
    angle_m += ((solver.x_p[p2] - C_x)*c1).cross(solver.v_p[p2]*c3)
    if C_x.x < 0.5:
      F_color[v3] = ti.Vector([ti.min(angle_m.norm(), 1.0), 0, 0])
    else:
      F_color[v3] = ti.Vector([0, 0, ti.min(angle_m.norm(), 1.0)])

# sdf = SphereSDF(dim=3, radius = 0.07, pos = ti.Vector([0.5, 0.35, 0.5]))
sdf = HandlerSDF(dim=3, pos=np.array([[0.3, 0.5, 0.3], [0.3, 0.5, 0.7], [0.7, 0.5, 0.3], [0.7, 0.5, 0.7]], dtype=np.float32), sphere_radius = 0.1)

simulator = AdaptiveGIMP(dim = 3,
                         level = 2,
                         sdf = sdf,
                         lag_force = get_force,
                         coarsest_size = 128,
                         n_particles  = NV,
                         particle_initializer = initialize,
                         grid_initializer = init_grid)

# exporter = MeshExporter("./results/cloth")

@ti.data_oriented
class DIYGUI:
    def __init__(self):
        self.window = ti.ui.Window("cloth", (1920, 1080), vsync=True, show_window=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.5, 1.0, 1.95)
        self.camera.lookat(0.5, 0.3, 0.5)
        self.camera.fov(55)

        self.bound_box = ti.Vector.field(3, ti.f32, shape=4)
        self.bound_box.from_numpy(np.array([[0.0, 0.0, 0.0], 
                                            [1.0, 0.0, 0.0], 
                                            [1.0, 0.0, 1.0], 
                                            [0.0, 0.0, 1.0]], dtype=np.float32))
        self.floor = ti.field(ti.i32, shape=6)
        self.floor.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
    
    @ti.kernel
    def calc_f_color(self, indices : ti.template(), pos : ti.template(), color : ti.template()):
      for v3 in range(NF):
        self.virtual_indices[v3*3+0] = v3*3+0
        self.virtual_indices[v3*3+1] = v3*3+1
        self.virtual_indices[v3*3+2] = v3*3+2

        self.virtual_points[v3*3+0] = pos[indices[v3*3+0]]
        self.virtual_points[v3*3+1] = pos[indices[v3*3+1]]
        self.virtual_points[v3*3+2] = pos[indices[v3*3+2]]

        self.virtual_colors[v3*3+0] = color[v3]
        self.virtual_colors[v3*3+1] = color[v3]
        self.virtual_colors[v3*3+2] = color[v3]

    def show(self, simulator, indices, pos, color):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.6, 0.6, 0.6))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        self.scene.mesh(self.bound_box, self.floor, color=(0.0, 0.0, 0.9), two_sided=True)

        if not hasattr(self, "virtual_indices"):
          self.virtual_indices = ti.field(ti.i32, shape=NF*3)
          self.virtual_points = ti.Vector.field(3, ti.f32, shape=NF*3)
          self.virtual_colors = ti.Vector.field(3, ti.f32, shape=NF*3)
        self.calc_f_color(indices, pos, color)
        simulator.sdf.render(self.scene)
        self.scene.mesh(self.virtual_points, self.virtual_indices, per_vertex_color=self.virtual_colors, two_sided=True)
        self.canvas.scene(self.scene)
        self.window.show()

def main():
    frame = 0
    gui = DIYGUI()
    while True:
        for _ in range(10):
            simulator.substep(dt)
        visualize(simulator)
        gui.show(simulator, indices, simulator.x_p, F_color)
        # if frame % 40 == 0: exporter.export(simulator.x_p.to_numpy(), F_verts.to_numpy().reshape(-1, 3), frame // 40)
        frame += 1

if __name__ == "__main__":
    main()
