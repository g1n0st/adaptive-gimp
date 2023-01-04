import taichi as ti
from utils import *

@ti.data_oriented
class AdaptiveGIMP:
  def __init__(self, level, grid_initializer, particle_initializer):
    # ----- adaptive grid data ------
    self.level = 4
    self.coarsest_size = 16
    self.coarsest_dx = 1 / self.coarsest_size
    self.finest_size = self.coarsest_size*(2**(self.level-1))
    self.boundary_gap = 2

    # NOTE: 0-coarsest, (level-1)-finest
    self.active_cell_mask = ti.field(ti.i32, shape=(self.level, self.finest_size, self.finest_size))
    self.active_node_mask = ti.field(ti.i32, shape=(self.finest_size+1, self.finest_size+1))
    # TODO(changyu): use dense grid for now
    self.ad_grid_v = ti.Vector.field(2, ti.f32, shape=(level, self.finest_size+1, self.finest_size+1))
    self.ad_grid_m = ti.field(ti.f32, shape=(level, self.finest_size+1, self.finest_size+1))


    # -------- particle data --------
    self.radius = 0.5 # Half-cell; this reflects what the radius is at the finest level of adaptivity
    self.p_mass = 1.0
    self.gravity = ti.Vector([0.0, -100.0])
    self.n_particles = 20000
    E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
    self.mu, self.la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

    self.x_p = ti.Vector.field(2, ti.f32, shape=self.n_particles)
    self.v_p = ti.Vector.field(2, ti.f32, shape=self.n_particles)
    self.F_p = ti.Matrix.field(2, 2, ti.f32, shape=self.n_particles)
    self.m_p = ti.field(ti.f32, shape=self.n_particles)
    self.fl_p = ti.field(ti.i32, shape=self.n_particles) # auxiliary data

    grid_initializer(self)
    self.mark_T_junction()
    self.mark_ghost()
    particle_initializer(self)

  @ti.func
  def _map(self, l, I): # map the level-l node to the finest node
    return I * (2**(self.level-1-l))

  @ti.func
  def activate_cell(self, l, I):
    ti.atomic_or(self.active_cell_mask[l, I // (2**(self.level-1-l))], ACTIVATED)

  def mark_T_junction(self):
    '''
          0x08
          |
    0x01 ------ 0x02
          |
          0x04
    '''
    REAL = 0xF0
    UP = 0x08
    DOWN = 0x04
    LEFT = 0x01
    RIGHT = 0x02
    @ti.kernel
    def _accumulate_mask(l : ti.template()):
      l_size = self.coarsest_size * (2**l)
      for i,j in ti.ndrange(l_size, l_size):
        if self.active_cell_mask[l, i,j] == ACTIVATED:
          for di,dj in ti.ndrange(2, 2):
            I_ = self._map(l, vec2(i+di,j+dj))
            ti.atomic_or(self.active_node_mask[I_], REAL)

            if i+di==0: ti.atomic_or(self.active_node_mask[I_], LEFT)
            if i+di==l_size: ti.atomic_or(self.active_node_mask[I_], RIGHT)
            if j+dj==0: ti.atomic_or(self.active_node_mask[I_], UP)
            if j+dj==l_size: ti.atomic_or(self.active_node_mask[I_], DOWN)

            if i+di-1>=0: ti.atomic_or(self.active_node_mask[self._map(l, vec2(i+di-1,j+dj))], RIGHT)
            if i+di+1<=l_size: ti.atomic_or(self.active_node_mask[self._map(l, vec2(i+di+1,j+dj))], LEFT)
            if j+dj-1>=0: ti.atomic_or(self.active_node_mask[self._map(l, vec2(i+di,j+dj-1))], DOWN)
            if j+dj+1<=l_size: ti.atomic_or(self.active_node_mask[self._map(l, vec2(i+di,j+dj+1))], UP)

    @ti.kernel
    def _mark_T_junction():
      for i,j in ti.ndrange(self.finest_size+1, self.finest_size+1):
        if self.active_node_mask[i, j] & 0xF0 > 0:
          if self.active_node_mask[i, j] & 0x0F == 0x0F: self.active_node_mask[i, j] = ACTIVATED
          elif self.active_node_mask[i, j] & 0x0F > 0: self.active_node_mask[i, j] = T_JUNCTION
        else: self.active_node_mask[i, j] = UNACTIVATED

    for l in range(self.level):
      _accumulate_mask(l)
    _mark_T_junction()
  
  def mark_ghost(self):
    dir = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    @ti.func
    def _heriarchical_check(l, I):
      res = False
      while l >= 0:
        if self.active_cell_mask[l, I] == ACTIVATED:
          res = True
          break
        l -= 1
        I //= 2
      return res

    @ti.kernel
    def _mark_ghost(l : ti.template()):
      l_size = self.coarsest_size * (2**l)
      for I in ti.grouped(ti.ndrange(l_size, l_size)):
        if self.active_cell_mask[l, I] == ACTIVATED:
          for _ in ti.static(range(4)):
            I1 = I+dir[_]
            if all(I1>=0 and I1<l_size) and self.active_cell_mask[l, I1] == UNACTIVATED \
              and _heriarchical_check(l, I1):
              tmp_l = l
              while self.active_cell_mask[tmp_l, I1] != ACTIVATED:
                ti.atomic_or(self.active_cell_mask[tmp_l, I1], GHOST)
                for cell_ in ti.grouped(ti.ndrange(2, 2)):
                  if self.active_node_mask[self._map(tmp_l, I1+cell_)] == UNACTIVATED:
                    ti.atomic_or(self.active_node_mask[self._map(tmp_l, I1+cell_)], GHOST)
                tmp_l -= 1
                I1 //= 2

    for l in reversed(range(self.level)):
      _mark_ghost(l)
  
  @ti.func
  def get_finest_level_near_particle(self, p):
    f_l = -1 # finest-level near the particle
    for l0 in range(self.level):
      l = self.level-l0-1 # reverse iteration
      dx = self.coarsest_dx / (2**l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      for dI in ti.grouped(ti.ndrange(2, 2)):
        if self.active_cell_mask[l, base+dI] == ACTIVATED:
          f_l = l
          break
      if f_l == l: break
    return f_l

  @ti.kernel
  def reinitialize_level(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
      self.ad_grid_m[l, I] = 0
      self.ad_grid_v[l, I].fill(0.0)
  
  @ti.kernel
  def p2g(self, dt0 : ti.f32):
    for p in range(self.n_particles):
      self.fl_p[p] = self.get_finest_level_near_particle(p)
      f_l = self.fl_p[p]

      dx = self.coarsest_dx / (2**f_l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      trilinear_coordinates = self.x_p[p] / dx - float(base+1)

      # compute stress
      stress = -dt0 * 0.25 / (self.radius * self.radius * dx) * get_stress(self.F_p[p], self.mu, self.la)

      for dI in ti.grouped(ti.ndrange(2, 2)):
        for cell_ in ti.grouped(ti.ndrange(2, 2)):
          weight, g_weight = get_linear_weight(self.radius, trilinear_coordinates, dI, cell_)
          self.ad_grid_m[f_l, base+dI+cell_] += weight * self.m_p[p]
          self.ad_grid_v[f_l, base+dI+cell_] += weight * (self.m_p[p] * self.v_p[p]) + stress @ g_weight
    
  @ti.kernel
  def g2p(self, dt0 : ti.f32):
    for p in range(self.n_particles):
      f_l = self.fl_p[p]
      dx = self.coarsest_dx / (2**f_l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      trilinear_coordinates = self.x_p[p] / dx - float(base+1)

      new_v = ti.Vector.zero(float, 2)
      new_G = ti.Matrix.zero(float, 2, 2)

      for dI in ti.grouped(ti.ndrange(2, 2)):
        for cell_ in ti.grouped(ti.ndrange(2, 2)):
          weight, g_weight = get_linear_weight(self.radius, trilinear_coordinates, dI, cell_)
          new_v += self.ad_grid_v[f_l, base+dI+cell_] * weight
          new_G += self.ad_grid_v[f_l, base+dI+cell_].outer_product(g_weight)
        
      self.v_p[p] = new_v
      self.x_p[p] += dt0 * self.v_p[p] # advection
      self.F_p[p] = (ti.Matrix.identity(float, 2) + dt0 * 0.25 / (self.radius * self.radius * dx) * new_G) @ self.F_p[p]

  @ti.kernel
  def grid_op(self, l : ti.template(), dt0 : ti.f32):
    l_size = self.coarsest_size * (2**l)+1
    for I in ti.grouped(ti.ndrange(l_size, l_size)):
      if self.ad_grid_m[l, I] > 0 and self.active_node_mask[self._map(l, I)] == ACTIVATED: # only on real degree-of-freedom
        self.ad_grid_v[l, I] /= self.ad_grid_m[l, I]
        self.ad_grid_v[l, I] += self.gravity * dt0
        for v in ti.static(range(2)):
          if self._map(l, I)[v] < self.boundary_gap * 2**(self.level-1) or \
             self._map(l, I)[v] > self.finest_size - self.boundary_gap * 2**(self.level-1):
            self.ad_grid_v[l, I][v] = 0
  
  @ti.kernel
  def grid_restriction(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
      if (self.active_node_mask[self._map(l, I)] == ACTIVATED or \
          self.active_node_mask[self._map(l, I)] == T_JUNCTION or \
          self.active_node_mask[self._map(l, I)] == GHOST):
        I2 = I * 2
        for dI in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
          if all(I2+dI>=0) and all(I2+dI <= l_size*2) and \
              self.active_node_mask[self._map(l+1, I2+dI)] == T_JUNCTION or \
              self.active_node_mask[self._map(l+1, I2+dI)] == GHOST or \
              (self.active_node_mask[self._map(l+1, I2+dI)] == ACTIVATED and all(dI==0)): # always count self
            weight = 0.5**float(ti.abs(dI).sum())
            self.ad_grid_m[l, I] += self.ad_grid_m[l+1, I2+dI] * weight
            self.ad_grid_v[l, I] += self.ad_grid_v[l+1, I2+dI] * weight

  @ti.kernel
  def grid_prolongation(self, l : ti.template()):
    l0 = l-1
    l0_size = self.coarsest_size * (2**l0)
    l_size = l0_size * 2
    for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
      if (self.active_node_mask[self._map(l, I)] == T_JUNCTION or \
          self.active_node_mask[self._map(l, I)] == GHOST or \
          self.active_node_mask[self._map(l, I)] == UNACTIVATED) or \
          all(I % 2 == 0): # non real-DOF should get value from real-DOF
        self.ad_grid_v[l, I].fill(0.0)

    for I in ti.grouped(ti.ndrange(l0_size+1, l0_size+1)):
      if (self.active_node_mask[self._map(l0, I)] == ACTIVATED or \
          self.active_node_mask[self._map(l0, I)] == T_JUNCTION or \
          self.active_node_mask[self._map(l0, I)] == GHOST):
        I2 = I * 2
        for dI in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
          if all(I2+dI>=0) and all(I2+dI <= l_size*2) and \
              self.active_node_mask[self._map(l, I2+dI)] == T_JUNCTION or \
              self.active_node_mask[self._map(l, I2+dI)] == GHOST or \
              (self.active_node_mask[self._map(l, I2+dI)] == ACTIVATED and all(dI==0)): # always count self
            weight = 0.5**float(ti.abs(dI).sum())
            self.ad_grid_v[l, I2+dI] += self.ad_grid_v[l0, I] * weight

  def substep(self, dt0):
    for l in range(self.level):
        self.reinitialize_level(l)
    
    self.p2g(dt0)
    
    for l in reversed(range(self.level)):
      if l != self.level - 1: self.grid_restriction(l)

    for l in range(self.level):
        self.grid_op(l, dt0)
    
    for l in range(self.level):
      if l != 0: self.grid_prolongation(l)

    self.g2p(dt0)