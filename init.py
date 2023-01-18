import taichi as ti

@ti.kernel
def initialize_particle(simulator : ti.template(), case : ti.template()):
  for i in range(simulator.n_particles):
    xy = ti.Vector([ti.random(), ti.random()])
    if i < simulator.n_particles // 2:
      simulator.x_p[i] = [xy[0] * 0.4 + 0.2, xy[1] * 0.4 + 0.1]
      simulator.v_p[i] = [0, 20.0]
    else:
      simulator.x_p[i] = [xy[0] * 0.4 + 0.4, xy[1] * 0.4 + 0.55]
      simulator.v_p[i] = [0, -20.0]

    simulator.F_p[i] = ti.Matrix.identity(ti.f32, 2)
    simulator.m_p[i] = simulator.p_mass

    if ti.static(case == 0):
      if all(0.2 <= xy <= 0.8):
        simulator.g_p[i] = 0
        simulator.c_p[i] = ti.Vector([0.93, 0.33, 0.23])
      elif all(0.1 <= xy <= 0.9):
        simulator.g_p[i] = 1
        simulator.c_p[i] = ti.Vector([0.95, 0.80, 0.43])
      elif all(0.03 <= xy <= 0.97):
        simulator.g_p[i] = 2
        simulator.c_p[i] = ti.Vector([0.80, 0.49, 0.80])
      else:
        simulator.g_p[i] = 3
        simulator.c_p[i] = ti.Vector([0.34, 0.51, 0.44])
    else:
      simulator.g_p[i] = -1
      simulator.c_p[i] = ti.Vector([0.93, 0.33, 0.23])

@ti.func
def initialize_mask1(simulator, I):
  simulator.activate_cell(simulator.level-1, I)
  return simulator.level-1

@ti.func
def initialize_mask2(simulator, I):
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
def initialize_mask3(simulator, I):
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
def initialize_mask4(simulator, I):
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
def initialize_mask5(simulator, I):
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