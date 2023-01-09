import taichi as ti
from utils import *

@ti.data_oriented
class GUI:
    def __init__(self, res=768):
      self.res = res
      self.bg_img = ti.Vector.field(3, ti.f32, shape=(res, res))
      self.window = ti.ui.Window('Adaptive GIMP', res = (res, res))
      self.canvas = self.window.get_canvas()

    @ti.kernel
    def visualize_mask(self, simulator : ti.template()):

      finest_size = simulator.coarsest_size * (2**(simulator.level-1))
      for i, j in ti.ndrange(self.res, self.res):
        self.bg_img[i, j].fill(0.0)
        for l in ti.static(range(simulator.level)):
          scale = self.res // (simulator.coarsest_size * (2**l))
          if simulator.active_cell_mask[l][i // scale, j // scale] == ACTIVATED: # normal activated cell
            self.bg_img[i, j] = ti.Vector([0.28, 0.68, 0.99])
            if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
              self.bg_img[i, j] = ti.Vector([0.18, 0.58, 0.88])
          elif simulator.active_cell_mask[l][i // scale, j // scale] == GHOST: # ghost cell
            self.bg_img[i, j] = ti.Vector([0, 0, 1.0])
            if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
              self.bg_img[i, j] = ti.Vector([0, 0, 0.77])
      
      for i, j in ti.ndrange(self.res, self.res):
        fscale = self.res // (finest_size)
        if i % fscale == 0 and j % fscale == 0:
          if simulator.active_node_mask[i // fscale, j // fscale] == ACTIVATED:
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
              if 0<=i+di < self.res and 0<=j+dj < self.res:
                self.bg_img[i+di, j+dj] = ti.Vector([1.0, 0.0, 0.0])
          elif simulator.active_node_mask[i // fscale, j // fscale] == T_JUNCTION:
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
              if 0<=i+di < self.res and 0<=j+dj < self.res:
                self.bg_img[i+di, j+dj] = ti.Vector([0.0, 1.0, 0.0])
          elif simulator.active_node_mask[i // fscale, j // fscale] == GHOST:
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
              if 0<=i+di < self.res and 0<=j+dj < self.res:
                self.bg_img[i+di, j+dj] = ti.Vector([0.0, 0.0, 0.55])
    
    @ti.kernel
    def visualize_sparse(self, simulator : ti.template()):
      finest_size = simulator.coarsest_size * (2**(simulator.level-1))
      for i, j in ti.ndrange(self.res, self.res):
        self.bg_img[i, j].fill(0.0)
        for l0 in ti.static(range(simulator.level)):
          l = ti.static(simulator.level-1-l0)
          if True:
            scale = self.res // (simulator.coarsest_size * (2**l))
            if ti.is_active(simulator.grid[l], ti.rescale_index(simulator.ad_grid_v[l], simulator.grid[l], [i // scale, j // scale])):
              if any(simulator.ad_grid_v[l][i // scale, j // scale] != 0):
                self.bg_img[i, j] = ti.Vector([0.93, 0.33, 0.23])
                if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
                  self.bg_img[i, j] = ti.Vector([0.83, 0.23, 0.13])
              else:
                self.bg_img[i, j] = ti.Vector([0.28, 0.68, 0.99])
                if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
                  self.bg_img[i, j] = ti.Vector([0.18, 0.58, 0.88])

    def show(self, simulator):
      self.visualize_mask(simulator)
      self.canvas.set_image(self.bg_img)
      self.canvas.circles(simulator.x_p, radius=0.006, color=(0.93, 0.33, 0.23))
      self.window.show()