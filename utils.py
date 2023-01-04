import taichi as ti

vec2 = ti.math.ivec2

UNACTIVATED = 0
ACTIVATED = 1
GHOST = 2
T_JUNCTION = 3

dt = 2e-5

@ti.func
def get_stress(F, mu, la):
  U, sig, V = ti.svd(F)
  J = F.determinant()
  stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
  return stress

@ti.func
def get_linear_weight(radius, trilinear_coordinates, dI, cell_):
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