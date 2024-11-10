
import numpy as np
import math
import matplotlib.pyplot as plt
import timeit


def graph_data(x, y, x_2, y_2):
    plt.figure()
    plt.scatter(x,y, color = 'purple')
    plt.plot(x_2,y_2, color = 'green')
    plt.xlabel('Time in minutes - normalized')
    plt.ylabel('p_NFKB -noramlized')
    #plt.title('Over expression of HER2. Block HER2 and IL1 1b (in silico)') #for 1b uncomment
    plt.title('Over expression of HER2. Block HER2 (in vivo) 1a')#for 1a uncomment
    plt.ylim([-0.05, 1.05])
    return plt.show()

# least squares
def gamma(a, b, c, d, t, n):
    result = 0
    for i in range(len(t)):
        result += (gompertz(a, b, c, d, t[i]) - n[i]) ** 2
    return result

# Gompertz function with d parameter
def gompertz(a, b, c, d, t):
    return a * np.exp(-b * np.exp(-c * t)) + d

def validate_interval(df, x0, x1):
    print(df(x0), df(x1))
    return df(x0) * df(x1) < 0

def bisection(df, x0, x1, max_iter=100, tol=1e-3):
    print("x0: ", x0, ", x1: ", x1)
    #if not validate_interval(df, x0, x1):
        #return
    for i in range(max_iter):
        approximation = x0 + (x1 - x0) / 2
        y = df(approximation)
        if -tol < y < tol:
            return approximation
        if validate_interval(df, x0, approximation):
            x1 = approximation
        else:
            x0 = approximation
    return approximation

# gradient of gamma
def grad(x0, norm_t, norm_n):
    a = x0[0].item()
    b = x0[1].item()
    c = x0[2].item()
    d = x0[3].item()

    partial_a = 0
    partial_b = 0
    partial_c = 0
    partial_d = 0
    
    
    #now, consolidated into one loop and clean it up:
    for i in range(len(norm_t)):
        #this is e^(-b*e^(-c*t))
        exp_term_no_coeff = np.exp(-b*np.exp(-c*norm_t[i]))
        #3 terms in side the sum
        inside_terms = (a*np.exp(-b*np.exp(-c*norm_t[i]))+d-norm_n[i])
        #e^((-c*t)-(b*e^(-c*t)))
        exp_term_minus_ct = np.exp((-c*norm_t[i])-b*np.exp(-c*norm_t[i]))
        #summations
        partial_a += (inside_terms)*(exp_term_no_coeff)
        partial_b += (inside_terms)*(-a)*(exp_term_minus_ct)
        partial_c += (inside_terms)*(exp_term_minus_ct)*(a*b*norm_t[i])
        partial_d += (inside_terms)
    
    g = np.zeros((4, 1))
    g[0] = partial_a
    g[1] = partial_b
    g[2] = partial_c
    g[3] = partial_d
    
    # g[0] = 4 * ((a - 4) ** 3)
    # g[1] = 2 * (b - 3)
    # g[2] = 16 * ((c + 5) ** 3)
    # g[3] = some other simplification?
    
    return g


def main():

    # below is the sample data from the 2024 lab supplemented by LaGrange interpolation
    data_1a = [(0, 0.8),
                (60, 0.8033),
                (120, 0.8067),
                (180, 0.81),
                (240, 0.8133),
                (300, 0.8167),
                (360, 0.82),
                (420, 0.6589),
                (480, 0.5222),
                (540, 0.41),
                (600, 0.3222),
                (660, 0.2589),
                (720, 0.22)]
    '''
    data_1b = [
                    ]
    '''

    '''
    1a with interpolation:
    [(0, 0.8),
        (60, 0.8033),
        (120, 0.8067),
        (180, 0.81),
        (240, 0.8133),
        (300, 0.8167),
        (360, 0.82),
        (420, 0.6589),
        (480, 0.5222),
        (540, 0.41),
        (600, 0.3222),
        (660, 0.2589),
        (720, 0.22)]
    '''
    '''
    1a wo int
    [
    (0,0.8),
    (180,0.81),
    (360,0.82),
    (540,0.41),
    (720,0.22)]
    '''


    # store x values (time) of the data into array t
    t = np.array([data[0] for data in data_1a])
    # store y values (number of cells) of the data into array n
    n = np.array([data[1] for data in data_1a])

    # normalization of the data
    min_n = float(min(n))
    max_n = float(max(n))
    norm_n = []  # this is the new list where the normalized data goes
    for i in range(len(n)):
        value = float((n[i] - min_n) / (max_n - min_n))
        norm_n.append(value)

    min_t = float(min(t))
    max_t = float(max(t))
    norm_t = []  # this is the new list where the normalized data goes
    for i in range(len(t)):
        value = float((t[i] - min_t) / (max_t - min_t))
        norm_t.append(value)

    # initialize variables
    iteration = 0
    x0 = np.array([-1, 55, 6, 1])
    x0 = np.reshape(x0, (4, 1))

    # calculate and solve the parameters using the gradient descent method
    while True:
        iteration += 1
        g = grad(x0, norm_t, norm_n)

        def d_ls_fun(z):
            dx0 = g
            x1 = x0 - z * dx0
            dx1 = grad(x1, norm_t, norm_n)
            return np.dot(dx0.T, dx1)

        step_size = bisection(d_ls_fun, -10, 10, 100, 1e-5)
        w = x0 - step_size * g
        
        print(f"iteration: {iteration}")
        print(f"partial a: {g[0]}, partial b:{g[1]}, partial c: {g[2]}, partial d: {g[3]}")

        print("termination condition: ", np.linalg.norm(w - x0) / np.linalg.norm(x0), np.linalg.norm(w - x0), np.linalg.norm(x0))

        # termination condition for the gradient descent algorithm
        if np.linalg.norm(w - x0) / np.linalg.norm(x0) < 1e-10:
            print(f"After {iteration} iterations: ")
            print(f"Parameter a is: {w[0]} and partial A is {g[0]}")
            print(f"Parameter b is: {w[1]} and partial B is {g[1]}")
            print(f"Parameter c is:4 {w[2]} and partial C is {g[2]}")
            print(f"Parameter d is: {w[3]} and partial D is {g[3]}")

            # Use the parameters found here for the function f(t). Get and store the output values for the graph
            x = np.linspace(0, 1, 100)
            f_t = []
            for i in range(len(x)):
                function_t = w[0] * np.exp(-w[1] * np.exp(-w[2] * x[i])) + w[3]
                f_t.append(function_t)
            graph_data(norm_t, norm_n, x, f_t)  # calls the graphing function and graphs
            break

        x0 = w


if __name__ == "__main__":
    # Measure execution time using timeit
    execution_time = timeit.timeit("main()", globals=globals(), number=1)
    
    # Print execution time
    print(f"Execution time: {execution_time:.4f} seconds")