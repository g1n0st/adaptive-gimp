import taichi as ti
ti.init(arch=ti.cpu)

# ----- adaptive grid data ------
level = 2
coarsest_size = 32
coarsest_dx = 1 / coarsest_size
finest_size = coarsest_size*(2**(level-1))

# TODO(changyu): use dense grid for now
# NOTE: 0-coarsest, (level-1)-finest 
active_mask = ti.field(ti.i32, shape=(level, finest_size, finest_size))
ad_grid_v = ti.Vector.field(2, ti.f32, shape=(level, finest_size+1, finest_size+1))
ad_grid_m = ti.field(ti.f32, shape=(level, finest_size+1, finest_size+1))

@ti.kernel
def initialize_mask():
  finest_size = coarsest_size * (2**(level-1))
  for i, j in ti.ndrange(finest_size, finest_size):
    # if j < finest_size / 2:
    if False: # FIXME(changyu): for test GIMP
      active_mask[0, i // 2, j // 2] = 1
    else:
      active_mask[1, i, j] = 1

initialize_mask()
# --------------------------------

dt = 1e-4
gravity = ti.Vector([0.0, -9.8])

# -------- particle data --------
radius = 0.5 # Half-cell; this reflects what the radius is at the finest level of adaptivity
p_mass = 1.0
n_particles = 10000
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
def initialize_particle():
  for i in range(n_particles):
    x_p[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.3]
    v_p[i] = [0, -1]
    F_p[i] = ti.Matrix.identity(ti.f32, 2)
    m_p[i] = p_mass

initialize_particle()
# --------------------------------

@ti.kernel
def reinitialize(l : ti.template()):
  l_size = coarsest_size * (2**l)
  for I in ti.grouped(ti.ndrange(l_size, l_size)):
    if active_mask[l, I] == 2:
      active_mask[l, I] = 0
    
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
      mx = 1.-bmax[v]
      mn = 1.-bmin[v]
    if mx < mn: 
      tmp = mx
      mx = mn
      mn = tmp
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
      if active_mask[l, base+dI] == 1:
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
      if active_mask[f_l, base+dI] == 0: active_mask[f_l, base+dI] = 2 # ghost cell, mark it 2
      for cell_ in ti.grouped(ti.ndrange(2, 2)):
        weight, g_weight = get_weight(trilinear_coordinates, dI, cell_)
        ad_grid_m[f_l, base+dI+cell_] += weight * m_p[p]
        ad_grid_v[f_l, base+dI+cell_] += weight * m_p[p] * v_p[p]
        # ad_grid_v[f_l, base+dI+cell_] += weight * (m_p[p] * v_p[p]) + stress @ g_weight

@ti.kernel
def grid_op(l : ti.template()):
  l_size = coarsest_size * (2**l)+1
  for I in ti.grouped(ti.ndrange(l_size, l_size)):
    if ad_grid_m[l, I] > 0:
      ad_grid_v[l, I] /= ad_grid_m[l, I]
      ad_grid_v[l, I] += gravity * dt
      I_ = I * (2**(level-1-l))
      for v in ti.static(range(2)):
        if I_[v] < 3 or I_[v] > finest_size - 3:
          ad_grid_v[l, I][v] = 0.0

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
      # F_p[p] = (ti.Matrix.identity(float, 2) + dt * 0.25 / (radius * radius * dx) * new_G) @ F_p[p]

@ti.kernel
def grid_accumulate(l : ti.template()):
  l_size = coarsest_size * (2**l)+1
  for I in ti.grouped(ti.ndrange(l_size, l_size)):
    pass

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
      if active_mask[l, i // scale, j // scale] == 1: # normal activated cell
        bg_img[i, j] = ti.Vector([0.28, 0.68, 0.99])
        if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
          bg_img[i, j] = ti.Vector([0.18, 0.58, 0.88])
      elif active_mask[l, i // scale, j // scale] == 2: # ghost cell
        bg_img[i, j] = ti.Vector([0, 0, 0.99])
        if i % scale == 0 or i % scale == scale-1 or j % scale == 0 or j % scale == scale-1:
          bg_img[i, j] = ti.Vector([0, 0, 0.77])

def show():
  visualize_mask()
  canvas.set_image(bg_img)
  canvas.circles(x_p, radius=0.006, color=(0.93, 0.33, 0.23))
  window.show()

# --------------------------------

while True:
  for l in range(level):
    reinitialize(l)
  p2g()
  
  '''
  for l in reversed(range(level)):
    if l != level - 1:
      grid_accumulate(l)
  '''

  for l in range(level):
    grid_op(l)

  g2p()

  show()