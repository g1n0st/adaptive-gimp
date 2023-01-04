import taichi as ti

@ti.kernel
def initialize_particle(simulator : ti.template()):
  for i in range(simulator.n_particles):
    simulator.x_p[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.4 + 0.3]
    simulator.v_p[i] = [0, -20.0]
    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
    simulator.m_p[i] = simulator.p_mass

@ti.kernel
def initialize_mask1(simulator : ti.template()):
  sz = simulator.finest_size
  for I in ti.grouped(ti.ndrange(sz, sz)):
    if I[0] < sz / 2:
      simulator.activate_cell(0, I)
    else: simulator.activate_cell(3, I)

@ti.kernel
def initialize_mask2(simulator : ti.template()):
  sz = simulator.finest_size
  for I in ti.grouped(ti.ndrange(sz, sz)):
    if I[0] < sz / 4:
      simulator.activate_cell(0, I)
    elif I[0] < sz / 2: simulator.activate_cell(1, I)
    else: simulator.activate_cell(2, I)

@ti.kernel
def initialize_mask3(simulator : ti.template()):
  sz = simulator.finest_size
  for I in ti.grouped(ti.ndrange(sz, sz)):
    if I[0] < sz / 2:
      if I[1] < sz / 2: simulator.activate_cell(1, I)
      else: simulator.activate_cell(2, I)
    else: simulator.activate_cell(3, I)

@ti.kernel
def initialize_mask4(simulator : ti.template()):
  sz = simulator.finest_size
  for I in ti.grouped(ti.ndrange(sz, sz)):
    if I[0] < sz / 2:
      if I[1] < sz / 4: simulator.activate_cell(0, I)
      elif I[1] < sz / 2: simulator.activate_cell(1, I)
      else: simulator.activate_cell(2, I)
    else: simulator.activate_cell(3, I)