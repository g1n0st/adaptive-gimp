import taichi as ti
from utils import *

@ti.data_oriented
class AdaptiveGIMP:
  def __init__(self, 
               dim, 
               level, coarsest_size,
               n_particles,
               particle_initializer, 
               grid_initializer = None):
    self.dim = dim
    # ----- adaptive grid data ------
    self.level = level
    self.coarsest_size = coarsest_size
    self.coarsest_dx = 1 / self.coarsest_size
    self.finest_size = self.coarsest_size*(2**(self.level-1))
    self.boundary_gap = 4

    # NOTE: 0-coarsest, (level-1)-finest
    self.axis = ti.ij if self.dim == 2 else ti.ijk

    self.node_mask_grid = ti.root.pointer(self.axis, (align_size(self.finest_size//4+1, 4), ) * self.dim)
    self.node_mask_block = self.node_mask_grid.dense(self.axis, 4)
    self.active_node_mask = ti.field(ti.i32)
    self.node_mask_block.place(self.active_node_mask)

    self.leaf_size = 4
    self.grid = []
    self.dense_block = []
    self.block = []
    self.active_cell_mask = []
    self.ad_grid_v = []
    self.ad_grid_m = []
    for l in range(level):
      l_size = coarsest_size * (2**l)
      leaf_size = 8
      self.grid.append(ti.root.pointer(self.axis, (align_size(l_size//leaf_size+1, 4), ) * self.dim))
      self.block.append(self.grid[l].bitmasked(self.axis, leaf_size))
      self.active_cell_mask.append(ti.field(int))
      self.ad_grid_v.append(ti.Vector.field(self.dim, float))
      self.ad_grid_m.append(ti.field(float))
      self.block[l].place(self.active_cell_mask[l], self.ad_grid_v[l], self.ad_grid_m[l])

    # -------- particle data --------
    self.radius = 0.5 # Half-cell; this reflects what the radius is at the finest level of adaptivity
    self.p_mass = 1.0
    self.gravity = ti.Vector([-100.0 if _ == 1 else 0.0 for _ in range(dim)])
    self.n_particles = n_particles
    E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
    self.mu, self.la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

    self.x_p = ti.Vector.field(dim, ti.f32, shape=self.n_particles)
    self.v_p = ti.Vector.field(dim, ti.f32, shape=self.n_particles)
    self.F_p = ti.Matrix.field(dim, dim, ti.f32, shape=self.n_particles)
    self.m_p = ti.field(ti.f32, shape=self.n_particles)
    self.fl_p = ti.field(ti.i32, shape=self.n_particles) # auxiliary data
    self.level_block = ti.root.dense(ti.i, self.level)
    self.pid = ti.field(int)
    self.level_block.dynamic(ti.j, self.n_particles, chunk_size = 16 * (2**self.dim)).place(self.pid)

    particle_initializer(self)
    self.grid_initializer = grid_initializer
    self.static_adaptivity = (grid_initializer != None)

  @ti.func
  def activate_cell(self, l : ti.template(), I):
    ti.atomic_or(self.active_cell_mask[l][I // (2**(self.level-1-l))], ACTIVATED)
  
  @ti.kernel
  def activate_cell_static(self):
    for p in range(self.n_particles):
      f_dx = self.coarsest_dx / (2**(self.level-1))
      f_base = (self.x_p[p] / f_dx + 0.5).cast(int) - 1

      for dI in ti.grouped(ti.ndrange(*(((-1, 3), ) * self.dim))):
        self.grid_initializer(self, (f_base+dI))

      level = self.grid_initializer(self, f_base)
      dx = self.coarsest_dx / (2**(level))
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
        self.grid_initializer(self, (base+dI) * 2**(self.level-1-level))

  @ti.func
  def _map(self, l, I): # map the level-l node to the finest node
    return I * (2**(self.level-1-l))
  
  @ti.func
  def is_valid(self, l, I): return (self.active_node_mask[self._map(l, I)] > 0)
  
  @ti.func
  def is_activated(self, l, I): return (self.active_node_mask[self._map(l, I)] & (ACTIVATED | T_JUNCTION)) == ACTIVATED

  @ti.func
  def is_T_junction(self, l, I): return (self.active_node_mask[self._map(l, I)] & T_JUNCTION) == T_JUNCTION
  
  @ti.func
  def is_ghost(self, l, I): return (self.active_node_mask[self._map(l, I)] & (ACTIVATED | GHOST | T_JUNCTION)) == GHOST
  
  @ti.kernel
  def _accumulate_mask(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    D = ti.Matrix.identity(int, self.dim)
    for I in ti.grouped(self.active_cell_mask[l]):
      if self.active_cell_mask[l][I] == ACTIVATED:
        for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
          ti.atomic_or(self.active_node_mask[self._map(l, I+dI)], ACTIVATED)

  def mark_activated(self):
    for l in range(self.level):
      self._accumulate_mask(l)
  
  @ti.func
  def _heriarchical_check(self, l, I):
    return ti.Vector([_ if (self.active_cell_mask[_][I // (2**ti.max(l-_, 0))] == ACTIVATED and _ <= l) \
      else -1 for _ in ti.static(range(0, self.level))], int).max()

  @ti.kernel
  def _mark_ghost_and_T_junction(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    D = ti.Matrix.identity(int, self.dim)
    for I in ti.grouped(self.active_cell_mask[l]):
      if self.active_cell_mask[l][I] == ACTIVATED:
        for _ in ti.static(range(self.dim * 2)):
          I1 = I+D[:, _//2]*(-1 if _%2 else 1)
          l_p = self._heriarchical_check(l, I1) # parent level
          if all(0<=I1<l_size) and self.active_cell_mask[l][I1] != ACTIVATED and l_p > -1:
            # check T_junction node
            for cell_ in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
              if cell_[_//2] == _%2:
                if any((I1+cell_) % (2**(l - l_p)) != 0):
                  ti.atomic_or(self.active_node_mask[self._map(l, I1+cell_)], T_JUNCTION)

            # check ghost node
            for l0 in ti.static(range(self.level)):
              if l_p < l0 <= l:
                I1_p = I1 // (2**ti.max(l-l0, 0))
                ti.atomic_or(self.active_cell_mask[l0][I1_p], GHOST)
                for cell_ in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
                  I1_pn = I1_p+cell_
                  if self.active_node_mask[self._map(l0, I1_pn)] == UNACTIVATED:
                    ti.atomic_or(self.active_node_mask[self._map(l0, I1_pn)], GHOST)

  def mark_ghost_and_T_junction(self):
    for l in reversed(range(self.level)):
      self._mark_ghost_and_T_junction(l)
  
  @ti.func
  def get_finest_level_near_particle(self, p):
    f_l = -1 # finest-level near the particle
    for l in ti.static(range(self.level)):
      dx = self.coarsest_dx / (2**l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
        if self.active_cell_mask[l][base+dI] == ACTIVATED: f_l = ti.max(f_l, l)
    return f_l

  @ti.kernel
  def reinitialize_level(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(self.ad_grid_m[l]):
      self.ad_grid_m[l][I] = 0
      self.ad_grid_v[l][I].fill(0.0)

  @ti.kernel
  def level_mapping(self):
    for p in range(self.n_particles):
      self.fl_p[p] = self.get_finest_level_near_particle(p)
      self.pid[self.fl_p[p]].append(p)
  
  @ti.func
  def get_weight_stencil(self, trilinear_coordinates):
    __weight = ti.Matrix.zero(float, ti.static((3 ** self.dim)), ti.static(self.dim+1))
    __conv = ti.Vector([3**_ for _ in ti.static(range(self.dim))])
    for dI in ti.static(ti.grouped(ti.ndrange(*((2, ) * self.dim)))):
      for cell_ in ti.static(ti.grouped(ti.ndrange(*((2, ) * self.dim)))):
        weight, g_weight = get_linear_weight(self.dim, self.radius, trilinear_coordinates, dI, cell_)
        __weight[(dI+cell_).dot(__conv), 0] += weight
        __weight[(dI+cell_).dot(__conv), 1:] += g_weight
    
    return __weight, __conv

  @ti.kernel
  def p2g(self, l : ti.template(), dt0 : ti.f32):
    for i in range(self.pid[l].length()):
      p = self.pid[l, i]

      dx = self.coarsest_dx / (2**l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      trilinear_coordinates = self.x_p[p] / dx - float(base+1)

      # compute stress
      stress = -dt0 * (0.5**self.dim) / ((self.radius**self.dim) * dx) * get_stress(self.dim, self.F_p[p], self.mu, self.la)
      
      __weight, __conv = self.get_weight_stencil(trilinear_coordinates)

      for dI in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
          tmp = __weight[dI.dot(__conv), :]
          weight, g_weight = tmp[0], tmp[1:]

          self.ad_grid_m[l][base+dI] += weight * self.m_p[p]
          self.ad_grid_v[l][base+dI] += weight * (self.m_p[p] * self.v_p[p]) + stress @ g_weight
    
  @ti.kernel
  def g2p(self, l : ti.template(), dt0 : ti.f32):
    for i in range(self.pid[l].length()):
      p = self.pid[l, i]

      dx = self.coarsest_dx / (2**l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      trilinear_coordinates = self.x_p[p] / dx - float(base+1)

      new_v = ti.Vector.zero(float, self.dim)
      new_G = ti.Matrix.zero(float, self.dim, self.dim)

      __weight, __conv = self.get_weight_stencil(trilinear_coordinates)

      for dI in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
          tmp = __weight[dI.dot(__conv), :]
          weight, g_weight = tmp[0], tmp[1:]
          new_v += self.ad_grid_v[l][base+dI] * weight
          new_G += self.ad_grid_v[l][base+dI].outer_product(g_weight)
        
      self.v_p[p] = new_v
      self.x_p[p] += dt0 * self.v_p[p] # advection
      self.F_p[p] = (ti.Matrix.identity(float, self.dim) + dt0 * (0.5**self.dim) / ((self.radius**self.dim) * dx) * new_G) @ self.F_p[p]

  @ti.kernel
  def grid_op(self, l : ti.template(), dt0 : ti.f32):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(self.ad_grid_v[l]):
      if all(0 <= I <= l_size) and self.ad_grid_m[l][I] > 0 and self.is_activated(l, I): # only on real degree-of-freedom
        self.ad_grid_v[l][I] /= self.ad_grid_m[l][I]
        self.ad_grid_v[l][I] += self.gravity * dt0
        for v in ti.static(range(self.dim)):
          if self._map(l, I)[v] < self.boundary_gap or \
             self._map(l, I)[v] > self.finest_size - self.boundary_gap:
            self.ad_grid_v[l][I][v] = 0

  @ti.kernel
  def grid_restriction(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(self.ad_grid_m[l]):
      if all(0 <= I <= l_size) and \
         ((self.is_activated(l, I) and all(I % 2 == 0)) or self.is_T_junction(l, I) or self.is_ghost(l, I)):
        for dI in ti.grouped(ti.ndrange(*(((-1, 2), ) * self.dim))):
          if all(0 <= I+dI <= l_size) and all ((I+dI) % 2 == 0):
            I0 = (I+dI)//2
            if self.is_valid(l-1, I0):
              weight = 0.5**float(ti.abs(dI).sum())
              self.ad_grid_m[l-1][I0] += self.ad_grid_m[l][I] * weight
              self.ad_grid_v[l-1][I0] += self.ad_grid_v[l][I] * weight

  @ti.kernel
  def grid_prolongation(self, l : ti.template()):
    l0 = ti.static(l-1)
    l0_size = self.coarsest_size * (2**l0)
    l_size = l0_size * 2
    for I in ti.grouped(self.ad_grid_v[l]):
      # non real-DOF should get value from real-DOF
      if all(0 <= I <= l_size) and (self.is_T_junction(l, I) or self.is_ghost(l, I) or all(I % 2 == 0)):
        self.ad_grid_v[l][I].fill(0.0)

    for I in ti.grouped(self.ad_grid_v[l0]):
      if all(0 <= I <= l0_size) and self.is_valid(l0, I):
        I2 = I * 2
        for dI in ti.grouped(ti.ndrange(*(((-1, 2), ) * self.dim))):
          if all(0 <= I2+dI <= l_size) and \
              (self.is_T_junction(l, I2+dI) or self.is_ghost(l, I2+dI) or \
              (self.is_activated(l, I2+dI) and all(dI==0))): # always count self
            weight = 0.5**float(ti.abs(dI).sum())
            self.ad_grid_v[l][I2+dI] += self.ad_grid_v[l0][I] * weight

  def substep(self, dt0):
    self.level_block.deactivate_all()
    self.node_mask_grid.deactivate_all()
    for l in range(self.level):
      self.grid[l].deactivate_all()
    
    if self.static_adaptivity:
      self.activate_cell_static()
    
    self.mark_activated()
    self.mark_ghost_and_T_junction()
    
    self.level_mapping()
    for l in range(self.level):
      self.p2g(l, dt0)

    for l in reversed(range(self.level)):
      if l != 0: self.grid_restriction(l)

    for l in range(self.level):
        self.grid_op(l, dt0)
    
    for l in range(self.level):
      if l != 0: self.grid_prolongation(l)

    for l in range(self.level):
      self.g2p(l, dt0)