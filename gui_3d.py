import taichi as ti
import numpy as np
from utils import *

@ti.data_oriented
class GUI3D:
    def __init__(self, simulator, res=768):
      self.res = res
      self.window = ti.ui.Window('Adaptive GIMP', res = (res, res))
      self.canvas = self.window.get_canvas()
      self.gui = self.window.get_gui()
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

      self.simulator = simulator
      self.line_vertex = ti.Vector.field(3, ti.f32, shape=simulator.n_particles * 24 * 10)
      self.line_count = ti.field(ti.i32, shape=())
      self.stencil = ti.field(ti.i32, shape=24)
      self.stencil.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, \
                                        0, 2, 1, 3, 4, 6, 5, 7, \
                                        0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32))

    @ti.kernel
    def _draw_lines(self, l : ti.template()):
      dx = self.simulator.coarsest_dx / (2 ** l)
      for I in ti.grouped(self.simulator.cell_mask[l]):
        if self.simulator.cell_mask[l][I] == ACTIVATED:
          index = ti.atomic_add(self.line_count[None], 24)
          idx = ti.Matrix.zero(ti.f32, 8, 3)
          conv = ti.Vector([1, 2, 4])
          for dI in ti.grouped(ti.ndrange(*((2, ) * 3))):
            idx[dI.dot(conv), :] = (I+dI) * dx
          for i in range(24):
            self.line_vertex[index+i] = idx[self.stencil[i], :] 

    
    def draw_lines(self):
      self.line_count[None] = 0
      for l in range(self.simulator.level):
        self._draw_lines(l)
    
    def show(self):
      self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
      self.scene.set_camera(self.camera)

      self.scene.ambient_light((0, 0, 0))

      self.scene.particles(self.simulator.x_p, per_vertex_color = self.simulator.c_p, radius=0.005)
      self.scene.mesh(self.bound_box, self.floor, color=(0.0, 0.0, 0.9), two_sided=True)
      self.simulator.sdf.render(self.scene)
      # self.draw_lines()
      # self.scene.lines(self.line_vertex, width = 0.001, vertex_count = self.line_count[None], color=(1.0, 1.0, 1.0))

      self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
      self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

      self.canvas.scene(self.scene)
      self.window.show()