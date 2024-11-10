"""
READ:
SeHwan Kim
11.09.2024


This code is for solving a second order ODE with the finite difference method (2nd order).
In this code, handling the Jacobian matrix (solved with Thomas algorithm) and implementation
of Newton's method is demonstrated.

This project was done in Fall of 2024 for my course work here at USC.
"""

import numpy as np

N = 800
b_c = 0.7

step_t = float((2*np.pi) / N)  # discretize the time
print(f"N = {N} and h = {step_t}")

# theta's element is theta(t) at each t_i, e.g. theta[0] = theta(t_0), theta[1] = theta(t_1), and so on.
# initial uniform guess for all theta(t_i's) set at 0.7
theta = np.ones(N+1) * b_c # N+1 array to hold the initial t_0 and t_N (t_N would be stored at N+1 element spot in the array)

# set b.c.
theta[0] = b_c
theta[-1] = b_c # theta[-1] is the last element, t_N.

# compute G(theta) - the LHS of the nonlinear system
def GofTheta(theta):
    # initialize the array to store G 
    G = np.zeros(N+1) #array of 0's, 1x101
    for i in range(1,N): #iterate from 1 to N-1: so that we consider t_0 and t_N, and not include t_-1 and t_N+1
        G[i] = (1/step_t**2)*(theta[i-1]-(2*theta[i])+theta[i+1])+np.sin(theta[i])

    return G

GofTheta(theta)

# set the Jacobian matrix
def CalcJacobianMatrix(theta):
    # 3 separate arrays to hold each sections of the tridiagonal matrix
    diag_J = np.zeros(N-1)
    upper_J = np.zeros(N-2)
    lower_J = np.zeros(N-2)

    for i in range(1,N):
        diag_J[i-1] = (-2/step_t**2)+np.cos(theta[i])
    for i in range(1,N-1):
        upper_J[i-1] = 1/step_t**2
        lower_J[i-1] = 1/step_t**2

    return diag_J, upper_J, lower_J

# solve for the delta^{k} using Thomas method/algorithm
def solveDelta(diag_J,upper_J,lower_J,RHS):
    N = len(diag_J)

    # forward elimination
    for i in range(1,N):
        factor = lower_J[i-1] / diag_J[i-1]
        diag_J[i] -= factor * upper_J[i-1]
        RHS[i] -= factor * RHS[i-1]

    # back sub
    delta = np.zeros(N)
    delta[-1] = RHS[-1] / diag_J[-1]

    for i in range(N-2,-1,-1): #start N-2, stop -1, step with -1 (i.e. backwards)
        delta[i] = (RHS[i]-(upper_J[i]*delta[i+1])) / diag_J[i]

    return delta

def CalcNewtonsMethod(theta, tol=1e-3,max_iter=10000):
    for iter in range(max_iter):
        G_funct = GofTheta(theta)
        # print(-1*G_funct)
        # set Jacobian
        diag_J, upper_J, lower_J = CalcJacobianMatrix(theta)
        # print(diag_J)
        # solve for delta using Thomas algo
        delta = solveDelta(diag_J,upper_J,lower_J,-1*G_funct[1:-1])
        # print(delta)
        # check for convergence by checking inf norm of delta less than tol
        if np.linalg.norm(delta, ord = np.inf)<=tol:
            print(f"Converged! Iteration#: {iter}")
            break
        if iter >= max_iter:
            print("Too many iterations")
            break
        # max_delta = np.max(delta)
        # print(f"Delta{iter}: max: {max_delta}")
        # update theta
        theta[1:-1] += delta
        # print(theta)

    return theta

solution = CalcNewtonsMethod(theta)

print(f"Found solution: {solution}")
np.savetxt(f"solution_N_{N}_h_{step_t:.2f}.csv", solution, delimiter=",", fmt="%.6f", header=f"solutions based on step size: {step_t} and N = {N}")