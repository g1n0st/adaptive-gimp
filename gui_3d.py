import taichi as ti
import numpy as np
from utils import *

@ti.data_oriented
class GUI3D:
    def __init__(self, res=768):
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
    
    def show(self, simulator):
      self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
      self.scene.set_camera(self.camera)

      self.scene.ambient_light((0, 0, 0))

      self.scene.particles(simulator.x_p, color = (0.5, 0.5, 0.5), radius=0.02)
      self.scene.mesh(self.bound_box, self.floor, color=(0.0, 0.0, 0.9), two_sided=True)

      self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
      self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

      self.canvas.scene(self.scene)
      self.window.show()