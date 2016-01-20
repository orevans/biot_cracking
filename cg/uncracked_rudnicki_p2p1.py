from dolfin import *
import numpy as np
from scipy.special import expi
import matplotlib.pyplot as plt

# Set parameters
B   = Constant(0.6)
q   = Constant(1.0)
G   = Constant(0.5)
K   = Constant(1.0)
c   = (3.*float(B)**2 + 3.*float(K) + 4.*float(G))/(3.*float(K) + 4.*float(G))
tol = 0.01 # Tolerance for specifying boundaries

# Define mesh and function spaces
mesh = Mesh("../mesh/thick_quarter_cylinder.xml")
U = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = MixedFunctionSpace([U, P])

# Define boundaries
class Outer(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]**2 + x[1]**2) > 8.5**2 - tol) and on_boundary

class Inner(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]**2 + x[1]**2) < .1**2 + DOLFIN_EPS) and on_boundary

class XAxis(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] < DOLFIN_EPS) and on_boundary

class YAxis(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < tol) and on_boundary

# Mark subdomains
out = Outer()
inn = Inner()
xaxis = XAxis()
yaxis = YAxis()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
out.mark(boundaries, 3)
inn.mark(boundaries, 4)
xaxis.mark(boundaries, 1)
yaxis.mark(boundaries, 2)

# Dirichlet boundary conditions
bc_x = DirichletBC(M.sub(0).sub(1), 0., boundaries, 1)
bc_y = DirichletBC(M.sub(0).sub(0), 0., boundaries, 2)
bcs  = [bc_x, bc_y]
ds = Measure("ds", subdomain_data=boundaries)

# Initial conditions
u_init = Constant((0.0, 0.0))
p_init = Constant(0.0)
u_prev = project(u_init, U)
p_prev = interpolate(p_init, P)

# Define trial and test functions
dv = TrialFunction(M)
(u_t, p_t) = TestFunctions(M)
vfunc = Function(M)
u, p = split(vfunc)

# Time-step
t = 0.0
dt = 0.1
T = 2*dt

# Visualize solution
f  = File("pressure.pvd")
fe = File("pressure_exact.pvd")

# Analytic solution
p_e = Function(P)
n = P.dim()
d = mesh.geometry().dim()
p_e_values = np.zeros(n)

dof_coords = P.dofmap().tabulate_all_coordinates(mesh)
dof_coords.resize((n,d))
dof_x = dof_coords[:,0]
dof_y = dof_coords[:,1]

while(t < T):
    # Update analytic solution
    p_e_values = -(1./(4.*np.pi))*expi(-c*(dof_x**2 + dof_y**2)/(4.*t))
    p_e.vector()[:] = p_e_values

    # Define stress tensor
    epsilon = sym(grad(u))
    epsilond = epsilon - tr(epsilon)/3.*Identity(2)
    sigma = 2.*G*epsilond + K*tr(epsilon)*Identity(2)
    epsilon_t = sym(grad(u_t))

    # Define facet normal and radius
    n = FacetNormal(mesh)
    r = Expression("pow(x[0]*x[0] + x[1]*x[1], 0.5)")

    # Weak form
    F = inner(sym(grad(u_t)), sigma)*dx - B*div(u_t)*p*dx + inner(u_t, n)*p*ds(3) + inner(u_t, n)*p*ds(4) 
    F += p_t*B*div(u - u_prev)*dx + p_t*(p - p_prev)*dx + dt*inner(grad(p_t), grad(p))*dx - dt*p_t/2./3.1415926538/r*ds(4) 

    # solve
    J = derivative(F, vfunc, dv)
    solve(F == 0, vfunc, bcs=bcs, J=J, solver_parameters={"newton_solver": {'absolute_tolerance': 1E-8, 'relative_tolerance': 1E-8, 'maximum_iterations': 25, 'relaxation_parameter': 1.0}})

    u, p = split(vfunc)

    # Print error (pressure)
    print "L2 error: %s" % assemble(inner(p - p_e, p - p_e)*dx)

    assign(u_prev, vfunc.sub(0))
    assign(p_prev, vfunc.sub(1))
    t += dt

f  << vfunc.sub(1)
fe << p_e

