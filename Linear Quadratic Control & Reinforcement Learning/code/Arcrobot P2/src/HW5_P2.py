
# coding: utf-8

# # Linearize the acrobot dynamics around the state x0
#     so that obtain the linear approximation

# In[3]:


import sympy
import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from collections import namedtuple
from sympy.abc import r, k, g, xi
from scipy.linalg import solve_continuous_are as solver


def Linearizing (r_const, g_const, xi_const, X_linearpoint):
    sympy.var('x0:4')
    X = sympy.Matrix(4,1,sympy.var('x0:4'))
    u = sympy.Matrix([sympy.var('u')])

    # inertia matrix
    M = sympy.Matrix ([[3 + 2*sympy.cos(x1), 1 + sympy.cos(x1)], [1 + sympy.cos(x1), 1]])
    
    # Coriolis, centripetal and gravitational forces
    c1 = x3*(2*x2+x3)*sympy.sin(x1)+2*g*sympy.sin(x0)+g*sympy.sin(x0+x1)
    c2 = -x2**2*sympy.sin(x1)+g*sympy.sin(x0+x1)
    
    # passive dynamics
    temp_a = sympy.zeros(4,4)
    temp_a[0:2, 0:2] = sympy.eye(2)
    temp_a[2:4, 2:4] = M.inv()
    a = temp_a * sympy.Matrix(4,1,[x2, x3, c1 - xi*x2, c2 - xi*x3])
    
    # control gain
    temp_b = sympy.zeros(4,4)
    temp_b[0:2, 0:2] = sympy.eye(2)
    temp_b[2:4, 2:4] = M.inv() 
    b = temp_b * sympy.Matrix(4,1, [0, 0, 0, 1])

    a = a.subs([(g, g_const), (xi, xi_const)])
    b = b.subs([(g, g_const), (xi, xi_const)])

    f = a + b*u #given the xi value and g value
    
    #linearization
    A = f.jacobian(X)
    B = f.jacobian(u)

    A0 = A.subs([(x0, X_linearpoint[0]),(x1, X_linearpoint[1]),(x2, X_linearpoint[2]),(x3, X_linearpoint[3])]) #given the equilibrium point
    B0 = B.subs([(x0, X_linearpoint[0]),(x1, X_linearpoint[1]),(x2, X_linearpoint[2]),(x3, X_linearpoint[3])]) #given the equilibrium point
    X_dot = A0 * X + B0 * u
    return a,b, A0, B0, X_dot


def Q_approximation (r_const, k_const):
    sympy.var('x0:4')
    X = sympy.Matrix(4,1,sympy.var('x0:4'))
    u = sympy.Matrix([sympy.var('u')])
    
    stage_cost =1- sympy.exp(k*sympy.cos(x0) + k*sympy.cos(x1) -2*k) + ((r**2)/2 * u*u)[0,0]
    mystage_cost = stage_cost.subs([(r,r_const), (k,k_const)])
    Q = sympy.hessian(mystage_cost, X)
    stage_cost0 = (0.5*X.T*Q*X + r**2*u**2/2) 
    return Q, stage_cost0
    
    
def LQR (Q, A0, B0, X_linearpoint):
    myQ = Q.subs([(x0,X_linearpoint[0]),(x1,X_linearpoint[1]),(x2,X_linearpoint[2]),(x3,X_linearpoint[3])])
    
    myQ_lqr = sympy.matrix2numpy(myQ).astype(float)
    A0_lqr = sympy.matrix2numpy(A0).astype(float)
    B0_lqr = sympy.matrix2numpy(B0).astype(float)
    
    M = solver(A0_lqr, B0_lqr, myQ_lqr, 1)
    return M

def nonlinear_system (y, t, a, b, my_pi):
    uu = my_pi(y[:, None])
    dydt = a.subs([(x0,y[0]),(x1,y[1]),(x2,y[2]),(x3,y[3])]) +    b.subs([(x0,y[0]),(x1,y[1]),(x2,y[2]),(x3,y[3])])*uu
    
    dydt = (sympy.matrix2numpy(dydt).astype(float)).squeeze()
    return dydt
print('Functions Imported')


# sympy.var('x0:4')
# X = sympy.Matrix(4,1,sympy.var('x0:4'))
# u = sympy.Matrix([sympy.var('u')])
# Consts = namedtuple('Consts', ['r_const', 'g_const', 'xi_const', 'k_const', 'X_linearpoint'])
# myconsts = Consts(r_const = 1, g_const = 9.8, xi_const = 1, k_const = 1, X_linearpoint = np.array([0,0,0,0]))

# a, b, A0, B0, X_dot = Linearizing (myconsts.r_const, myconsts.g_const, myconsts.xi_const, myconsts.X_linearpoint)

# Q, stage_cost0 = Q_approximation (myconsts.r_const, myconsts.k_const)
# M = LQR (Q, A0, B0, myconsts.X_linearpoint)
# my_pi = lambda X: - 1/myconsts.r_const * B0.T * M * X



# # construction on nonlinearized system
# y0 = myconsts.X_linearpoint  #np.array([0,0,0,0]) #initial condition
# t = np.linspace(0, 10, 101)
# my_pi = lambda X: - 1/myconsts.r_const * B0.T * M * X
# sol = odeint(nonlinear_system, y0, t, args=(a, b, my_pi))
