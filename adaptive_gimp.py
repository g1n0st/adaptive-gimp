import taichi as ti
from utils import *

ENABLE_AFFINE = False

@ti.data_oriented
class AdaptiveGIMP:
  def __init__(self, 
               dim,
               sdf,
               level, coarsest_size,
               n_particles,
               particle_initializer, 
               grid_initializer = None,
               lag_force = None):
    self.time = 0.0
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
    self.node_mask = ti.field(ti.i32)
    self.node_mask_block.place(self.node_mask)

    self.leaf_size = 4
    self.grid = []
    self.block = []
    self.cell_mask = []
    self.ad_grid_v = []
    self.ad_grid_m = []
    self.pid = []
    for l in range(level):
      l_size = coarsest_size * (2**l)
      self.grid.append(ti.root.pointer(self.axis, (align_size(l_size//self.leaf_size+1, 4), ) * self.dim))
      self.block.append(self.grid[l].bitmasked(self.axis, self.leaf_size))
      self.cell_mask.append(ti.field(int))
      self.ad_grid_v.append(ti.Vector.field(self.dim, float))
      self.ad_grid_m.append(ti.field(float))
      self.block[l].place(self.cell_mask[l], self.ad_grid_v[l], self.ad_grid_m[l])

      self.pid.append(ti.field(int))
      self.grid[l].dynamic(ti.axes(self.dim), n_particles, chunk_size = 16 * (2**self.dim)).place(self.pid[l])

    # -------- particle data --------
    self.radius = 0.5 # Half-cell; this reflects what the radius is at the finest level of adaptivity
    self.p_mass = 1.0
    self.gravity = ti.Vector([-9.8 if _ == 1 else 0.0 for _ in range(dim)])
    self.n_particles = n_particles
    E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
    self.mu, self.la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

    self.x_p = ti.Vector.field(dim, ti.f32, shape=self.n_particles)
    self.v_p = ti.Vector.field(dim, ti.f32, shape=self.n_particles)
    self.f_p = ti.Vector.field(dim, ti.f32, shape=self.n_particles)
    self.F_p = ti.Matrix.field(dim, dim, ti.f32, shape=self.n_particles)
    self.C_p = ti.Matrix.field(dim, dim, ti.f32, shape=self.n_particles) # affine in APIC
    self.m_p = ti.field(ti.f32, shape=self.n_particles)
    self.c_p = ti.Vector.field(3, ti.f32, shape=self.n_particles) # particle color
    self.g_p = ti.field(ti.i32, shape=self.n_particles) # auxiliary data: prescribed grid level for particles
    self.fl_p = ti.field(ti.i32, shape=self.n_particles) # auxiliary data: finest level near particles

    particle_initializer(self)
    self.grid_initializer = grid_initializer
    self.static_adaptivity = (grid_initializer != None)
    self.lag_force = lag_force
    self.sdf = sdf
    self.friction = 0.0

  @ti.func
  def activate_cell(self, l : ti.template(), I):
    ti.atomic_or(self.cell_mask[l][I // (2**(self.level-1-l))], ACTIVATED)

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

  @ti.kernel
  def activate_cell_dynamic(self):
    for p in range(self.n_particles):
      dx = self.coarsest_dx / (2**self.g_p[p])
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1

      for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
        for L in ti.static(range(self.level)): 
          if L == self.g_p[p]:
            ti.atomic_or(self.cell_mask[L][base+dI], ACTIVATED)

  @ti.kernel
  def _mark_hier_(self, l : ti.template(), verbose : ti.template()):
    for I in ti.grouped(self.cell_mask[l]):
      if (self.cell_mask[l][I] & ACTIVATED):
        l0 = -1
        for l_ in ti.static(range(l)):
          if (self.cell_mask[l_][I // (2**(l-l_))] & ACTIVATED):
            l0 = ti.max(l0, l_)
        if l0 != -1:
          for l_ in ti.static(range(self.level)):
            if l0 <= l_ < l:
              ti.atomic_or(self.cell_mask[l_][I // (2**ti.max(l-l_, 0))], CULLING)

  @ti.kernel
  def _split_hier_(self, l : ti.template()):
    for I in ti.grouped(self.cell_mask[l]):
      if (self.cell_mask[l][I] & CULLING):
        for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
          ti.atomic_or(self.cell_mask[ti.static(l+1)][I*2+dI], ACTIVATED)

        self.cell_mask[l][I] = 0

  def culling_hierarchy_duplicate(self):
    for l in reversed(range(self.level)):
      if l != 0: self._mark_hier_(l, 0)
    for l in range(self.level):
      if l != self.level-1: self._split_hier_(l)

  @ti.func
  def _map(self, l, I): # map the level-l node to the finest node
    return I * (2**(self.level-1-l))

  @ti.func
  def is_valid(self, l, I): return (self.node_mask[self._map(l, I)] > 0)

  @ti.func
  def is_activated(self, l, I): return (self.node_mask[self._map(l, I)] & (ACTIVATED | T_JUNCTION)) == ACTIVATED

  @ti.func
  def is_T_junction(self, l, I): return (self.node_mask[self._map(l, I)] & T_JUNCTION) == T_JUNCTION

  @ti.func
  def is_ghost(self, l, I): return (self.node_mask[self._map(l, I)] & (ACTIVATED | GHOST | T_JUNCTION)) == GHOST

  @ti.kernel
  def _activate_mask(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    D = ti.Matrix.identity(int, self.dim)
    for I in ti.grouped(self.cell_mask[l]):
      if self.cell_mask[l][I] == ACTIVATED:
        for dI in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
          ti.atomic_or(self.node_mask[self._map(l, I+dI)], ACTIVATED)

  def mark_activated(self):
    for l in range(self.level):
      self._activate_mask(l)

  @ti.func
  def _heriarchical_check(self, l, I):
    return ti.Vector([_ if (self.cell_mask[_][I // (2**ti.max(l-_, 0))] == ACTIVATED and _ <= l) \
      else -1 for _ in ti.static(range(0, self.level))], int).max()

  @ti.kernel
  def _mark_ghost_and_T_junction(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    D = ti.Matrix.identity(int, self.dim)
    for I in ti.grouped(self.cell_mask[l]):
      if self.cell_mask[l][I] == ACTIVATED:
        for _ in ti.static(range(self.dim * 2)):
          I1 = I+D[:, _//2]*(-1 if _%2 else 1)
          l_p = self._heriarchical_check(l, I1) # parent level
          if all(0<=I1<l_size) and self.cell_mask[l][I1] != ACTIVATED and l_p > -1:
            # check T_junction node
            for cell_ in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
              if cell_[_//2] == _%2:
                if any((I1+cell_) % (2**(l - l_p)) != 0):
                  ti.atomic_or(self.node_mask[self._map(l, I1+cell_)], T_JUNCTION)

            # check ghost node
            for l0 in ti.static(range(self.level)):
              if l_p < l0 <= l:
                I1_p = I1 // (2**ti.max(l-l0, 0))
                ti.atomic_or(self.cell_mask[l0][I1_p], GHOST)
                for cell_ in ti.grouped(ti.ndrange(*((2, ) * self.dim))):
                  if self.node_mask[self._map(l0, I1_p+cell_)] == UNACTIVATED:
                    ti.atomic_or(self.node_mask[self._map(l0, I1_p+cell_)], GHOST)

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
        if self.cell_mask[l][base+dI] == ACTIVATED: f_l = ti.max(f_l, l)
    return f_l

  @ti.kernel
  def reinitialize_level(self, l : ti.template()):
    l_size = self.coarsest_size * (2**l)
    for I in ti.grouped(self.ad_grid_m[l]):
      self.ad_grid_m[l][I] = 0
      self.ad_grid_v[l][I].fill(0.0)

  @ti.kernel
  def level_mapping(self):
    ti.loop_config(block_dim=16)
    for p in range(self.n_particles):
      self.fl_p[p] = self.get_finest_level_near_particle(p)
      for l in ti.static(range(self.level)):
          if self.fl_p[p] == l:
            dx = self.coarsest_dx / (2**l)
            base = (self.x_p[p] / dx + 0.5).cast(int) - 1
            base_pid = ti.rescale_index(self.ad_grid_m[l], self.pid[l].parent(2), base)
            ti.append(self.pid[l].parent(), base_pid, p)

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
    for I in ti.grouped(self.pid[l]):
      p = self.pid[l][I]

      dx = self.coarsest_dx / (2**l)
      base = (self.x_p[p] / dx + 0.5).cast(int) - 1
      trilinear_coordinates = self.x_p[p] / dx - float(base+1)

      stress = ti.Matrix.zero(float, self.dim, self.dim)
      if ti.static(self.lag_force == None):
        # compute stress
        stress = -dt0 * (0.5**self.dim) / ((self.radius**self.dim) * dx) * get_stress(self.dim, self.F_p[p], self.mu, self.la)
        self.f_p[p].fill(0.0)

      __weight, __conv = self.get_weight_stencil(trilinear_coordinates)

      for dI in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
          tmp = __weight[dI.dot(__conv), :]
          weight, g_weight = tmp[0], tmp[1:]
          dpos = dI.cast(float) - (trilinear_coordinates+1)

          self.ad_grid_m[l][base+dI] += weight * self.m_p[p]
          self.ad_grid_v[l][base+dI] += weight * self.m_p[p] * (self.v_p[p] + self.C_p[p] @ dpos) + stress @ g_weight + weight * self.f_p[p] * dt0

  @ti.kernel
  def g2p(self, l : ti.template(), dt0 : ti.f32):
    for I in ti.grouped(self.pid[l]):
      p = self.pid[l][I]

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
      if ti.static(ENABLE_AFFINE):
        self.C_p[p] = new_G
      self.x_p[p] += dt0 * self.v_p[p] # advection
      if ti.static(self.lag_force == None):
        self.F_p[p] = (ti.Matrix.identity(float, self.dim) + dt0 * (0.5**self.dim) / ((self.radius**self.dim) * dx) * new_G) @ self.F_p[p]

  @ti.kernel
  def grid_op(self, l : ti.template(), dt0 : ti.f32):
    l_size = self.coarsest_size * (2**l)
    dx = self.coarsest_dx / (2**l)
    for I in ti.grouped(self.ad_grid_v[l]):
      if all(0 <= I <= l_size) and self.ad_grid_m[l][I] > 0 and self.is_activated(l, I): # only on real degree-of-freedom
        vel = self.ad_grid_v[l][I]
        vel /= self.ad_grid_m[l][I]
        vel += self.gravity * dt0

        # boundary condition
        for v in ti.static(range(self.dim)):
          if self._map(l, I)[v] < self.boundary_gap or \
             self._map(l, I)[v] > self.finest_size - self.boundary_gap:
            vel[v] = 0

          pos = (I + 0.5) * dx
          fixed, inside, dotnv, diff_vel, n = self.sdf.check(pos, vel)
          if inside:
            if fixed: vel.fill(0.0)
            else:
              dotnv_frac = dotnv * (1.0 - self.friction)
              vel += diff_vel * self.friction + n * dotnv_frac

        self.ad_grid_v[l][I] = vel

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
    self.sdf.update(self.time)
    self.node_mask_grid.deactivate_all()
    for l in range(self.level):
      self.grid[l].deactivate_all()

    if self.static_adaptivity:
      self.activate_cell_static()
    else:
      self.activate_cell_dynamic()
      self.culling_hierarchy_duplicate() # ensure the hierarchy will not be activated duplicately

    self.mark_activated()
    self.mark_ghost_and_T_junction()

    self.level_mapping()

    if self.lag_force != None:
      self.lag_force(self)

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
    
    self.time += dt0