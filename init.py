import taichi as ti

@ti.kernel
def initialize_particle(simulator : ti.template()):
  for i in range(simulator.n_particles):
    simulator.x_p[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.4 + 0.3]
    simulator.v_p[i] = [0, -20.0]
    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
    simulator.m_p[i] = simulator.p_mass

@ti.func
def initialize_mask0(simulator, I):
  simulator.activate_cell(simulator.level-1, I)
  return simulator.level-1

@ti.func
def initialize_mask1(simulator, I):
  sz = simulator.finest_size
  L = -1
  if I[0] < sz / 2 and all(8 <= I < sz - 8):
    L = 0
    simulator.activate_cell(0, I)
  else: 
    L = 3
    simulator.activate_cell(3, I)
  return L

@ti.func
def initialize_mask2(simulator, I):
  sz = simulator.finest_size
  L = -1
  if I[0] < sz / 4 and all(8 <= I < sz - 8):
    L = 0
    simulator.activate_cell(0, I)
  elif I[0] < sz / 2 and all(8 <= I < sz - 8): 
    L = 1
    simulator.activate_cell(1, I)
  else:
    L = 2
    simulator.activate_cell(2, I)
  return L

@ti.func
def initialize_mask3(simulator, I):
  sz = simulator.finest_size
  L = 3
  if I[0] < sz / 2 and all(8 <= I < sz - 8):
    if I[1] < sz / 2: 
      L = 1
      simulator.activate_cell(1, I)
    else:
      L = 2
      simulator.activate_cell(2, I)
  elif all(0 <= I < sz):
    L = 3
    simulator.activate_cell(3, I)
  return L

@ti.func
def initialize_mask4(simulator, I):
  sz = simulator.finest_size
  L = 3
  if I[0] < sz / 2 and all(8 <= I < sz - 8):
    if I[1] < sz / 4: 
      L = 0
      simulator.activate_cell(0, I)
    elif I[1] < sz / 2: 
      L = 1
      simulator.activate_cell(1, I)
    else: 
      L = 2
      simulator.activate_cell(2, I)
  elif all(0 <= I < sz): 
    L = 3
    simulator.activate_cell(3, I)
  return L