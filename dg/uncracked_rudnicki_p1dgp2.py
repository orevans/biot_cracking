from dolfin import *
import numpy as np
from scipy.special import expi
import matplotlib.pyplot as plt

# Set parameters
B   = Constant(.6)
q   = Constant(1.)
G   = Constant(.5)
K   = Constant(1.)
c   = (3.*float(B)**2 + 3.*float(K) + 4.*float(G))/(3.*float(K) + 4.*float(G))
s   = Constant(4.) # Stabilization coefficient
tol = 0.01 # Tolerance for specifying boundaries

# Define mesh and function spaces
mesh = Mesh("../mesh/thick_quarter_cylinder.xml")
U = VectorFunctionSpace(mesh, "DG", 1)  # P1-DG
P = FunctionSpace(mesh, "CG", 2)        # P2
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

ds = Measure("ds")[boundaries]

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

    # Define facet normal, cell averages and radius
    n = FacetNormal(mesh)
    r = Expression("pow(x[0]*x[0] + x[1]*x[1], 0.5)")
    h = CellVolume(mesh)
    havg = .5*(h('+') + h('-'))

    # Define stress tensor
    epsilon  = sym(grad(u))
    epsilond = epsilon - tr(epsilon)/3.*Identity(2)
    sigma    = 2.*G*epsilond + K*tr(epsilon)*Identity(2)
    
    epsilon_t  = sym(grad(u_t))
    epsilond_t = epsilon_t - tr(epsilon_t)/3.*Identity(2)
    sigma_t    = 2.*G*epsilond_t + K*tr(epsilon_t)*Identity(2)
    
    sigma_t_n   = dot(avg(sigma_t), n('+'))
    sigma_t_n_d = dot(sigma_t, n)
    sigma_n     = dot(avg(sigma), n('+'))
    sigma_n_d   = dot(sigma, n)

    # Stabilization tensors
    epsilon_stab  = outer(jump(u), n('+'))
    epsilond_stab = epsilon_stab - tr(epsilon_stab)/3.*Identity(2)
    sigma_stab    = 2.*G('+')*epsilon_stab + K('+')*tr(epsilon_stab)*Identity(2)

    # Weak form
    F = inner(epsilon_t, sigma)*dx + B*inner(u_t, grad(p))*dx \
        - inner(sigma_t_n, jump(u))*dS - inner(jump(u_t), sigma_n)*dS \
        + (s('+')/havg)*inner(outer(jump(u_t), n('+')), sigma_stab)*dS \
        - sigma_t_n_d[1]*u[1]*ds(2) - u_t[1]*sigma_n_d[1]*ds(2) \
        + (s/h)*u_t[1]*(4.*G*u[1]/3. + K*u[1]/3.)*ds(2) \
        - sigma_t_n_d[0]*u[0]*ds(1) - u_t[0]*sigma_n_d[0]*ds(1) \
        + (s/h)*u_t[0]*(4.*G*u[0]/3. + K*u[0]/3.)*ds(1)

    F += -inner(grad(p_t), B*(u - u_prev))*dx + p_t*B*inner((u - u_prev), n)*ds(4) \
         + p_t*B*inner((u - u_prev), n)*ds(3) + p_t*(p - p_prev)*dx \
         + dt*inner(grad(p_t), grad(p))*dx - dt*p_t/2./3.1415926538/r*ds(4)

    # solve
    J = derivative(F, vfunc, dv)
    problem = NonlinearVariationalProblem(F, vfunc, [], J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = "mumps"
    solver.solve()

    # Print error (pressure)
    print "L2 error: %s" % assemble(inner(p - p_e, p - p_e)*dx)

    assign(u_prev, vfunc.sub(0))
    assign(p_prev, vfunc.sub(1))
    t += dt

f  << vfunc.sub(1)
fe << p_e

