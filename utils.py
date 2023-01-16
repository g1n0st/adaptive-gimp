import taichi as ti

vec2 = ti.math.ivec2

UNACTIVATED = 0
ACTIVATED = 1 << 0
GHOST = 1 << 1
T_JUNCTION = 1 << 2

@ti.func
def get_stress(dim : ti.template(), F, mu, la):
  U, sig, V = ti.svd(F)
  J = F.determinant()
  stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose() + ti.Matrix.identity(float, dim) * la * J * (J - 1)
  return stress

@ti.func
def get_linear_weight(dim : ti.template(), radius, trilinear_coordinates, dI, cell_):
  bmin = ti.math.clamp(trilinear_coordinates+(1-dI)-radius, 0.0, 1.0)
  bmax = ti.math.clamp(trilinear_coordinates+(1-dI)+radius, 0.0, 1.0)
  w = ti.Vector.zero(float, dim)
  g_w = ti.Vector.zero(float, dim)
  for v in ti.static(range(dim)):
    mx, mn=0.0, 0.0
    if cell_[v]:
      mx = bmax[v]
      mn = bmin[v]
    else: 
      mx = 1.-bmin[v]
      mn = 1.-bmax[v]
    w[v] = mx**2-mn**2
    g_w[v] = (mx-mn) if cell_[v] else (mn-mx)

  if ti.static(dim == 2):
    weight = w[0] * w[1] / (4.0 * (radius*2.0)**2)
    g_weight = ti.Vector([g_w[0] * w[1], w[0] * g_w[1]]) / (2.0 * (radius*2.0)**2)
    return weight, g_weight
  else:
    weight = w[0] * w[1] * w[2] / (8.0 * (radius*2.0)**3)
    g_weight = ti.Vector([g_w[0] * w[1] * w[2], w[0] * g_w[1] * w[2], w[0] * w[1] * g_w[2]]) / (4.0 * (radius*2.0)**3)
    return weight, g_weight

def align_size(x, align):
  return (x+(align-1))&~(align-1)