import numpy as np
def function_matrix(xy):
    x, y = xy
    return [x**2+y**2-2,
            x-y]

def jacobian_matrix(xy):
    x, y = xy
    return [[2*x,2*y],
            [1,-1]]

def iterative_newton(fun, x_init, jacobian):
    max_iter = 50
    epsilon = 1e-8

    x_last = x_init

    for k in range(max_iter):
        # Solve J(xn)*( xn+1 - xn ) = -F(xn):
        J = np.array(jacobian(x_last))
        F = np.array(fun(x_last))

        diff = np.linalg.solve( J, -F )
        x_last = x_last + diff

        # Stop condition:
        if np.linalg.norm(diff) < epsilon:
            print('convergence!, nre iter:', k )
            break

    else: # only if the for loop end 'naturally'
        print('not converged')

    return x_last

x_sol = iterative_newton(function_matrix, [-1.1,-1.2], jacobian_matrix)
print('Solutions:', x_sol )
print('F(sol)', function_matrix(x_sol) )