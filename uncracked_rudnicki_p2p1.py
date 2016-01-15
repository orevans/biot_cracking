xfrom dolfin import *
import matplotlib.pyplot as plt

# Parameters
B = Constant(0.6)
q = Constant(1.0)
G = Constant(0.5)
K = Constant(1.0)

# Define mesh and function spaces
mesh = Mesh("thick_quarter_cylinder.xml")
U = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = MixedFunctionSpace([U, P])

# Compute soltuion
while(t < T):
    theta = Constant(0.5)
    ptheta = theta*p + (1. - theta)*p_prev
    utheta = theta*u + (1. - theta)*u_prev

    # Define stress tensor
    epsilon = sym(grad(u))
    epsilond = epsilon - tr(epsilon)/3.*Identity(2)
    sigma = 2.*G*epsilond + K*tr(epsilon)*Identity(2)
    epsilon_t = sym(grad(u_t))

    # Facet normal and radius
    n = FacetNormal(mesh)
    r = Expression("x[0]*x[0] + x[1]*x[1]")

    # Weak form
    F = inner(sym(grad(u_t)), sigma)*dx - B*div(u_t)*p*dx + inner(u_t, n)*p*ds(1) + inner(u_t, n)*p*ds(2) 
    F += p_t*B*div(u - u_prev)*dx + p_t*(p - p_prev)*dx + dt*inner(grad(p_t), grad(ptheta))*dx - dt*p_t*(5./r)*ds(0) 

    # Solve
    J = derivative(F, vfunc, dv)
    solve(F == 0, vfunc, bcs=bcs, J=J, solver_parameters={"newton_solver": {'absolute_tolerance': 1E-8, 'relative_tolerance': 1E-8, 'maximum_iterations': 25, 'relaxation_parameter': 1.0}})

    f << vfunc.sub(1)

    u, p = split(vfunc)
    assign(u_prev, vfunc.sub(0))
    assign(p_prev, vfunc.sub(1))
    t += dt
