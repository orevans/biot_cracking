from dolfin import *
import matplotlib.pyplot as plt

# parameters
B = Constant(0.6)
q = Constant(1.0)
G = Constant(0.5)
K = Constant(1.0)

# mesh + function spaces
mesh = Mesh("thick_quarter_cylinder.xml")
U = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = MixedFunctionSpace([U, P])
#b_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

class Outer(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]*x[0] + x[1]*x[1]), 8.5*8.5)

class Inne(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]*x[0] + x[1]*x[1]), .1*.1)

class XAxis(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.)

class YAxis(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.)

out = Outer()
inn = Inne()
xaxis = XAxis()
yaxis = YAxis()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
out.mark(boundaries, 1)
inn.mark(boundaries, 2)
xaxis.mark(boundaries, 3)
yaxis.mark(boundaries, 4)

# bcs
bc_x = DirichletBC(M.sub(0).sub(0), 0., boundaries, 3)
bc_y = DirichletBC(M.sub(0).sub(1), 0., boundaries, 4)
bcs  = [bc_x, bc_y]
ds = Measure("ds")[boundaries]

# ics
u_init = Constant((0.0, 0.0))
p_init = Constant(0.0)
u_prev = project(u_init, U)
p_prev = interpolate(p_init, P)

# trial + test functions
dv = TrialFunction(M)
(u_t, p_t) = TestFunctions(M)
vfunc = Function(M)
u, p = split(vfunc)

# time
t = 0
dt = 0.1
T = 100*dt

f = File("pressure.pvd")

# stress
#epsilon = sym(grad(u))
#epsilond = epsilon - tr(epsilon)/3.*Identity(2)
#sigma = 2.*G*epsilond + K*tr(epsilon)*Identity(2)
#epsilon_t = sym(grad(u_t))


while(t < T):
    theta = Constant(0.5)
    ptheta = theta*p + (1. - theta)*p_prev
    utheta = theta*u + (1. - theta)*u_prev

#    ds_in = ds

    epsilon = sym(grad(u))
    epsilond = epsilon - tr(epsilon)/3.*Identity(2)
    sigma = 2.*G*epsilond + K*tr(epsilon)*Identity(2)
    epsilon_t = sym(grad(u_t))

    n = FacetNormal(mesh)
    r = Expression("x[0]*x[0] + x[1]*x[1]")

    F = inner(sym(grad(u_t)), sigma)*dx - B*div(u_t)*p*dx + inner(u_t, n)*p*ds(1) + inner(u_t, n)*p*ds(2) 
    F += p_t*B*div(u - u_prev)*dx + p_t*(p - p_prev)*dx + dt*inner(grad(p_t), grad(ptheta))*dx - dt*p_t*(5./r)*ds(0) #- dt*p_t*(5./r)*ds(0)

    # solve
    J = derivative(F, vfunc, dv)
    solve(F == 0, vfunc, bcs=bcs, J=J, solver_parameters={"newton_solver": {'absolute_tolerance': 1E-8, 'relative_tolerance': 1E-8, 'maximum_iterations': 25, 'relaxation_parameter': 1.0}})

#    solve(F == 0, vfunc, bcs=bcs, J=J, solver_parameters={"nonlinear_solver": "snes"})

    f << vfunc.sub(1)

    u, p = split(vfunc)
    assign(u_prev, vfunc.sub(0))
    assign(p_prev, vfunc.sub(1))

    t += dt
