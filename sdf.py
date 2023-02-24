import taichi as ti

@ti.data_oriented
class SDF:
    def __init__(self, dim, fixed=True):
        self.dim = dim
        self.vel = ti.Vector.field(dim, float, shape=())
        self.fixed = ti.field(int, shape=())
        self.fixed[None] = fixed

    @ti.func
    def check(self, pos, vel):
        phi = self.dist(pos)
        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, self.dim)
        n = ti.Vector.zero(ti.f32, self.dim)
        if phi < 0.0:
            n = self.normal(pos)
            diff_vel = self.vel[None] - vel
            dotnv = n.dot(diff_vel)
            if dotnv > 0.0 or self.fixed[None]:
                inside = True
        
        return self.fixed[None], inside, dotnv, diff_vel, n

    def update(self, t):
        pass

    def render(self, scene):
        pass

    @ti.func
    def dist(self, pos):
        return 1.0
    
    @ti.func
    def normal(self, pos):
        return ti.Vector.zero(ti.f32, self.dim)

    def log(self, gui):
        pass

@ti.data_oriented
class HandlerSDF(SDF):
    def __init__(self, dim, pos, sphere_radius = 0.013):
        super().__init__(dim, fixed=True)
        self.sphere_pos = ti.Vector.field(self.dim, float, shape=pos.shape[0])
        self.sphere_pos.from_numpy(pos)
        self.sphere_radius = sphere_radius

    @ti.kernel
    def update(self, t : ti.f32):
        pass

    @ti.func
    def dist(self, pos): # Function computing the signed distance field
        dist = 1e5
        for i in range(self.sphere_pos.shape[0]):
            dist = min((pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius, dist)
        return dist

    @ti.func
    def normal(self, pos): # Function computing the gradient of signed distance field
        dist = 1e5
        normal = ti.Vector.zero(ti.f32, self.dim)
        for i in range(self.sphere_pos.shape[0]):
            dist0 = (pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius
            if dist0 < dist:
                dist = dist0
                normal = (pos - self.sphere_pos[0]).normalized(1e-9)
        return normal
    
    def render(self, scene):
        scene.particles(self.sphere_pos, self.sphere_radius, color = (1, 0, 0))

    def log(self, gui):
        pass

@ti.data_oriented
class MixedSDF(SDF):
    def __init__(self, dim, sdf_a, sdf_b):
        super().__init__(dim, fixed=False)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
    
    def update(self, t):
        self.sdf_a.update(t)
        self.sdf_b.update(t)
    
    def render(self, scene):
        self.sdf_a.render(scene)
        self.sdf_b.render(scene)

    @ti.func
    def dist(self, pos):
        phi_a = self.sdf_a.dist(pos)
        phi_b = self.sdf_b.dist(pos)
        return phi_a if phi_a < phi_b else phi_b
    
    @ti.func
    def check(self, pos, vel):
        phi_a = self.sdf_a.dist(pos)
        phi_b = self.sdf_b.dist(pos)

        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, self.dim)
        n = ti.Vector.zero(ti.f32, self.dim)
        fixed = False
        fixed_0 = False
        fixed_1 = False

        if phi_a < phi_b:
            fixed_0, inside, dotnv, diff_vel, n = self.sdf_a.check(pos, vel)
        else:
            fixed_1, inside, dotnv, diff_vel, n = self.sdf_b.check(pos, vel)

        if phi_a < 0.0 and fixed_0: fixed = True
        if phi_b < 0.0 and fixed_1: fixed = True
        
        return fixed, inside, dotnv, diff_vel, n
    
    def log(self, gui):
        self.sdf_a.log(gui)
        self.sdf_b.log(gui)