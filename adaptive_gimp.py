import taichi as ti
import time
ti.init(arch=ti.cpu)

vec2 = ti.math.ivec2

# ----- adaptive grid data ------
UNACTIVATED = 0
ACTIVATED = -1
GHOST = -2
T_JUNCTION = -3

level = 2
coarsest_size = 32
coarsest_dx = 1 / coarsest_size
finest_size = coarsest_size*(2**(level-1))

# NOTE: 0-coarsest, (level-1)-finest
active_cell_mask = ti.field(ti.i32, shape=(level, finest_size, finest_size))
active_node_mask = ti.field(ti.i32, shape=(finest_size+1, finest_size+1))

@ti.func
def _map(l, I): # map the level-l node to the finest node
  return I * (2**(level-1-l))

# TODO(changyu): use dense grid for now
ad_grid_v = ti.Vector.field(2, ti.f32, shape=(level, finest_size+1, finest_size+1))
ad_grid_m = ti.field(ti.f32, shape=(level, finest_size+1, finest_size+1))

@ti.kernel
def initialize_mask():
  for i, j in ti.ndrange(finest_size, finest_size):
    if i >= finest_size / 2:
      active_cell_mask[0, i // 2, j // 2] = ACTIVATED
    else:
      active_cell_mask[1, i, j] = ACTIVATED

initialize_mask()

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
def accumulate_mask(l : ti.template()):
  l_size = coarsest_size * (2**l)
  for i,j in ti.ndrange(l_size, l_size):
    if active_cell_mask[l, i,j] == ACTIVATED:
      for di,dj in ti.ndrange(2, 2):
        I_ = _map(l, vec2(i+di,j+dj))
        ti.atomic_or(active_node_mask[I_], REAL)
        if i+di==0: ti.atomic_or(active_node_mask[I_], LEFT)
        if i+di==l_size: ti.atomic_or(active_node_mask[I_], RIGHT)
        if j+dj==0: ti.atomic_or(active_node_mask[I_], UP)
        if j+dj==l_size: ti.atomic_or(active_node_mask[I_], DOWN)

        if i+di-1>=0: ti.atomic_or(active_node_mask[_map(l, vec2(i+di-1,j+dj))], RIGHT)
        if i+di+1<=l_size: ti.atomic_or(active_node_mask[_map(l, vec2(i+di+1,j+dj))], LEFT)
        if j+dj-1>=0: ti.atomic_or(active_node_mask[_map(l, vec2(i+di,j+dj-1))], DOWN)
        if j+dj+1<=l_size: ti.atomic_or(active_node_mask[_map(l, vec2(i+di,j+dj+1))], UP)

@ti.kernel
def mark_mask():
  for i,j in ti.ndrange(finest_size+1, finest_size+1):
    if active_node_mask[i, j] & 0xF0 > 0:
      if active_node_mask[i, j] & 0x0F == 0x0F: active_node_mask[i, j] = ACTIVATED
      elif active_node_mask[i, j] & 0x0F > 0: active_node_mask[i, j] = T_JUNCTION
    else: active_node_mask[i, j] = UNACTIVATED

for l in range(level):
  accumulate_mask(l)
mark_mask()
# --------------------------------

dt = 1e-4
gravity = ti.Vector([0.0, -9.8])

# -------- particle data --------
radius = 0.5 # Half-cell; this reflects what the radius is at the finest level of adaptivity
p_mass = 1.0
n_particles = 20000
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu, la = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
x_p = ti.Vector.field(2, ti.f32, shape=n_particles)
v_p = ti.Vector.field(2, ti.f32, shape=n_particles)
F_p = ti.Matrix.field(2, 2, ti.f32, shape=n_particles)
m_p = ti.field(ti.f32, shape=n_particles)

# auxiliary data
fl_p = ti.field(ti.i32, shape=n_particles)

@ti.kernel
def reinitialize():
  for I in ti.grouped(ti.ndrange(finest_size+1, finest_size+1)):
    if active_node_mask[I] == GHOST:
      active_node_mask[I] = UNACTIVATED

@ti.kernel
def initialize_particle():
  for i in range(n_particles):
    x_p[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.2 + 0.3]
    v_p[i] = [0, -20.0]
    F_p[i] = ti.Matrix.identity(ti.f32, 2)
    m_p[i] = p_mass

initialize_particle()
# --------------------------------

@ti.kernel
def reinitialize_level(l : ti.template()):
  l_size = coarsest_size * (2**l)
  for I in ti.grouped(ti.ndrange(l_size, l_size)):
    if active_cell_mask[l, I] == GHOST:
      active_cell_mask[l, I] = UNACTIVATED
    
  for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
    ad_grid_m[l, I] = 0
    ad_grid_v[l, I].fill(0.0)

@ti.func
def get_weight(trilinear_coordinates, dI, cell_):
  bmin = ti.math.clamp(trilinear_coordinates+(1-dI)-radius, 0.0, 1.0)
  bmax = ti.math.clamp(trilinear_coordinates+(1-dI)+radius, 0.0, 1.0)
  w = ti.math.vec2(0.0)
  g_w = ti.math.vec2(0.0)
  for v in ti.static(range(2)):
    mx, mn=0.0, 0.0
    if cell_[v]:
      mx = bmax[v]
      mn = bmin[v]
    else: 
      mx = 1.-bmin[v]
      mn = 1.-bmax[v]
    w[v] = mx**2-mn**2
    g_w[v] = (mx-mn) if cell_[v] else (mn-mx)

  weight = w[0] * w[1] / (4.0 * (radius*2.0)**2)
  g_weight = ti.Vector([g_w[0] * w[1], w[0] * g_w[1]]) / (2.0 * (radius*2.0)**2)
  return weight, g_weight

@ti.func
def get_stress(F):
  U, sig, V = ti.svd(F)
  J = F.determinant()
  stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
  return stress

@ti.func
def get_finest_level_near_particle(p):
  f_l = -1 # finest-level near the particle
  for l0 in range(level):
    l = level-l0-1 # reverse iteration
    dx = coarsest_dx / (2**l)
    base = (x_p[p] / dx + 0.5).cast(int) - 1
    for dI in ti.grouped(ti.ndrange(2, 2)):
      if active_cell_mask[l, base+dI] == ACTIVATED:
        f_l = l
        break
    if f_l == l: break
  return f_l

@ti.kernel
def p2g():
  for p in range(n_particles):
    fl_p[p] = get_finest_level_near_particle(p)
    f_l = fl_p[p]

    dx = coarsest_dx / (2**f_l)
    base = (x_p[p] / dx + 0.5).cast(int) - 1
    trilinear_coordinates = x_p[p] / dx - float(base+1)

    # compute stress
    stress = -dt * 0.25 / (radius * radius * dx) * get_stress(F_p[p])

    for dI in ti.grouped(ti.ndrange(2, 2)):
      if active_cell_mask[f_l, base+dI] == UNACTIVATED: 
        active_cell_mask[f_l, base+dI] = GHOST # ghost cell
      for cell_ in ti.grouped(ti.ndrange(2, 2)):
        if active_node_mask[_map(f_l, base+dI+cell_)] == UNACTIVATED:
            active_node_mask[_map(f_l, base+dI+cell_)] = GHOST # ghost node
        weight, g_weight = get_weight(trilinear_coordinates, dI, cell_)
        ad_grid_m[f_l, base+dI+cell_] += weight * m_p[p]
        # ad_grid_v[f_l, base+dI+cell_] += weight * m_p[p] * v_p[p]
        ad_grid_v[f_l, base+dI+cell_] += weight * (m_p[p] * v_p[p]) + stress @ g_weight

@ti.kernel
def grid_op(l : ti.template()):
  l_size = coarsest_size * (2**l)+1
  for I in ti.grouped(ti.ndrange(l_size, l_size)):
    if ad_grid_m[l, I] > 0 and active_node_mask[_map(l, I)] == ACTIVATED: # only on real degree-of-freedom
      ad_grid_v[l, I] /= ad_grid_m[l, I]
      ad_grid_v[l, I] += gravity * dt
      for v in ti.static(range(2)):
        if _map(l, I)[v] < 4 or _map(l, I)[v] > finest_size - 4:
          ad_grid_v[l, I][v] = -ad_grid_v[l, I][v]

@ti.kernel
def g2p():
  for p in range(n_particles):
    f_l = fl_p[p]
    dx = coarsest_dx / (2**f_l)
    base = (x_p[p] / dx + 0.5).cast(int) - 1
    trilinear_coordinates = x_p[p] / dx - float(base+1)

    new_v = ti.Vector.zero(float, 2)
    new_G = ti.Matrix.zero(float, 2, 2)

    for dI in ti.grouped(ti.ndrange(2, 2)):
      for cell_ in ti.grouped(ti.ndrange(2, 2)):
        weight, g_weight = get_weight(trilinear_coordinates, dI, cell_)
        new_v += ad_grid_v[f_l, base+dI+cell_] * weight
        new_G += ad_grid_v[f_l, base+dI+cell_].outer_product(g_weight)
      
    v_p[p] = new_v
    x_p[p] += dt * v_p[p] # advection
    F_p[p] = (ti.Matrix.identity(float, 2) + dt * 0.25 / (radius * radius * dx) * new_G) @ F_p[p]

@ti.kernel
def grid_restriction(l : ti.template()):
  l_size = coarsest_size * (2**l)
  for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
    if active_node_mask[_map(l, I)] == ACTIVATED or active_node_mask[_map(l, I)] == T_JUNCTION:
      I2 = I * 2
      for dI in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
        if all(I2+dI>=0) and all(I2+dI <= l_size*2) and \
            active_node_mask[_map(l+1, I2+dI)] == T_JUNCTION or \
            active_node_mask[_map(l+1, I2+dI)] == GHOST or \
            (active_node_mask[_map(l+1, I2+dI)] == ACTIVATED and all(dI==0)): # always count self
          weight = 0.5**float(ti.abs(dI).sum())
          ad_grid_m[l, I] += ad_grid_m[l+1, I2+dI] * weight
          ad_grid_v[l, I] += ad_grid_v[l+1, I2+dI] * weight

@ti.kernel
def grid_prolongation(l : ti.template()):
  l0 = l-1
  l0_size = coarsest_size * (2**l0)
  l_size = l0_size * 2
  for I in ti.grouped(ti.ndrange(l_size+1, l_size+1)):
    #                                                                                         FIXME(changyu): here is still a hack!
    if active_node_mask[_map(l, I)] == T_JUNCTION or active_node_mask[_map(l, I)] == GHOST or all(I % 2 == 0): # non real-DOF should get value from real-DOF
      ad_grid_v[l, I].fill(0.0)

  for I in ti.grouped(ti.ndrange(l0_size+1, l0_size+1)):
    if active_node_mask[_map(l0, I)] == ACTIVATED or active_node_mask[_map(l0, I)] == T_JUNCTION:
      I2 = I * 2
      for dI in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
        if all(I2+dI>=0) and all(I2+dI <= l_size*2) and \
            active_node_mask[_map(l, I2+dI)] == T_JUNCTION or \
            active_node_mask[_map(l, I2+dI)] == GHOST or \
            (active_node_mask[_map(l, I2+dI)] == ACTIVATED and all(dI==0)): # always count self
          weight = 0.5**float(ti.abs(dI).sum())
          ad_grid_v[l, I2+dI] += ad_grid_v[l0, I] * weight


# ------- visualization -------
res = 640
bg_img = ti.Vector.field(3, ti.f32, shape=(res, res))
window = ti.ui.Window('Adaptive GIMP', res = (res, res))
canvas = window.get_canvas()

@ti.kernel
def visualize_mask():
  for i, j in ti.ndrange(res, res):
    bg_img[i, j].fill(0.0)
    for l in range(level):
      scale = res // (coarsest_size * (2**l))
      if active_cell_mask[l, i // scale, j // scale] == ACTIVATED: # normal activated cell
        bg_img[i, j] = ti.Vector([0.28, 0.68, 0.99])
        if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
          bg_img[i, j] = ti.Vector([0.18, 0.58, 0.88])
      elif active_cell_mask[l, i // scale, j // scale] == GHOST: # ghost cell
        bg_img[i, j] = ti.Vector([0, 0, 0.77])
        if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
          bg_img[i, j] = ti.Vector([0, 0, 0.55])
  
  for i, j in ti.ndrange(res, res):
    fscale = res // (finest_size)
    if i % fscale == 0 and j % fscale == 0:
      if active_node_mask[i // fscale, j // fscale] == ACTIVATED:
        for di, dj in ti.ndrange(3, 3):
          if 0<=i+di-1 < res and 0<=j+dj-1 < res:
            bg_img[i+di-1, j+dj-1] = ti.Vector([1.0, 0.0, 0.0])
      elif active_node_mask[i // fscale, j // fscale] == T_JUNCTION:
        for di, dj in ti.ndrange(3, 3):
          if 0<=i+di-1 < res and 0<=j+dj-1 < res:
            bg_img[i+di-1, j+dj-1] = ti.Vector([0.0, 1.0, 0.0])
      elif active_node_mask[i // fscale, j // fscale] == GHOST:
        for di, dj in ti.ndrange(3, 3):
          if 0<=i+di-1 < res and 0<=j+dj-1 < res:
            bg_img[i+di-1, j+dj-1] = ti.Vector([0.0, 0.0, 1.0])


def show():
  visualize_mask()
  canvas.set_image(bg_img)
  canvas.circles(x_p, radius=0.006, color=(0.93, 0.33, 0.23))
  window.show()

# --------------------------------

def substep():
  reinitialize()
  for l in range(level):
      reinitialize_level(l)
  
  p2g()
  
  for l in reversed(range(level)):
    if l != level - 1: grid_restriction(l)

  for l in range(level):
      grid_op(l)
  
  for l in range(level):
    if l != 0: grid_prolongation(l)

  g2p()

while True:
  for i in range(20):
    substep()

  show()
  # time.sleep(50)