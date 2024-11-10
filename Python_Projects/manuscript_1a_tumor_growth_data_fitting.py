import numpy as np
import math
import matplotlib.pyplot as plt
import timeit
import csv

# graphs my results
def graphData(date, ko_y, fit_x, fit_y):
    plt.figure()
    # scatter is the lab data
    plt.scatter(date, ko_y, color = 'red', label='y1: IL2 KO')
    plt.plot(fit_x, fit_y, color = 'purple', label='y2: IL2 Fit')
    plt.legend()
    plt.xlabel('Time - Days')
    plt.ylabel('Tumor size')
    return plt.show()

def validate_interval(df, x0, x1):
    #print(df(x0), df(x1))
    return df(x0) * df(x1) < 0

def y1(a,b,t):
    return a*t+b

def y2(a,b,c,t):
    return a*((t/40)**(b-1))*((1-t/40)**(c-1))

def bisection(df, x0, x1, max_iter=100, tol=1e-5):
    # print("x0: ", x0, ", x1: ", x1)
    # if not validate_interval(df, x0, x1):
        #return
    for i in range(max_iter):
        approximation = x0 + (x1 - x0) / 2
        y = df(approximation)
        if abs(y)<tol:
            return approximation
        if validate_interval(df, x0, approximation):
            x1 = approximation
        else:
            x0 = approximation
    return approximation

    # gradient of phi - returns 3x1 vector with partials calculated

def grad(x_k, date_x, ko_y):
    # calls scalar value of x_k vector's elements
    a = x_k[0].item()
    b = x_k[1].item()
    c = x_k[2].item()

    partial_a = 0
    partial_b = 0
    partial_c = 0
    '''
    # for y_1
    for i in range(len(date_x)):
        # summations
        partial_a += 2*(a*date_x[i]+b - wt_y[i])*date_x[i]
        partial_b += 2*(a*date_x[i]+b - wt_y[i])
    
    # create array of column vectors, 3x1, initialized with elements=0 to store partial values 
    grad_result = np.zeros((2, 1))
    # store partial a,b,c values into array g, which is a 3x1 column vector
    grad_result[0] = partial_a
    grad_result[1] = partial_b
    
    return grad_result
    '''
    # for y_2
    for i in range(len(date_x)):
        common_term = 2*(y2(a,b,c,date_x[i])-ko_y[i])
        partial_a += (common_term)*((date_x[i]/40)**(b-1))*((1-(date_x[i]/40))**(c-1))
        partial_b += (common_term)*(a*(1-(date_x[i]/40))**(c-1))*((date_x[i]/40)**(b-1))*(np.log(date_x[i]/40))
        partial_c += (common_term)*(a*(date_x[i]/40)**(b-1))*((1-(date_x[i]/40))**(c-1))*(np.log(1-(date_x[i]/40)))

    grad_result  = np.zeros((3,1))
    grad_result[0] = partial_a
    grad_result[1] = partial_b
    grad_result[2] = partial_c

    return grad_result

def main():
    # import CSV file
    average_data_csv = '/Users/sehwankim/Documents/USC/RESEARCH/Potential_New_Hexin_Chen/average_values_1a.csv'

    # list for the data points
    date = [] # column 1 of the CSV file
    avg_wt = [] # column 2 of the CSV file
    avg_ko = [] # column 3 of the CSV file

    # Import CSV data raw into the lists:
    with open(average_data_csv, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)# skip row 0 = "dates" and "prices"
        # Iterate over each row in the CSV reader object
        for row in csv_reader:
            date.append(float(row[0]))
            avg_wt.append(float(row[1]))
            avg_ko.append(float(row[2]))

    # initialize variables such as a, b, and c
    iteration = 0
    # array to hold the initial values - put in the initial value below
    x_k = np.array([0, 3, 8])
    # transpose the array "x_k", 1x3, above to vector 3x1
    x_k = np.reshape(x_k, (3, 1))

    # calculate and solve the parameters using the gradient descent method
    while True:
        iteration += 1
        g_k = grad(x_k, date, avg_ko)

        def d_ls_fun(z):
            dx_k = g_k
            x_k1 = x_k - z * dx_k
            dx_k1 = grad(x_k1, date, avg_ko)
            return np.dot(dx_k.T, dx_k1) # dot the two gradients to approximate the derivative of gamma of step size (i.e. argmin function)

        # find the zeros of the approximation of the optimal step size function
        step_size = bisection(d_ls_fun,-10,10)

        # use the found step size and x_k and g_k to update to x_{k+1}
        x_next = x_k - step_size * g_k
        
        print(f"iteration: {iteration}")
        print(f"partial a: {g_k[0]} and partial b:{g_k[1]} and partial c:{g_k[2]}")

        # termination condition
        if np.linalg.norm(x_next - x_k) / np.linalg.norm(x_k) < 1e-10:
            print(f"After {iteration} number of iterations: ")
            print(f"Parameter a is: {x_next[0]} and partial A is {g_k[0]}")
            print(f"Parameter b is: {x_next[1]} and partial B is {g_k[1]}")
            print(f"Parameter c is: {x_next[2]} and partial B is {g_k[2]}")

            # Use the parameters found for the function f(t). Get and store the output values for the graph
            x = np.linspace(0, 40, 100)
            f_t = []
            for t in x:
                function_t = y2(x_next[0],x_next[1],x_next[2],t)
                f_t.append(function_t)
            graphData(date, avg_ko, x, f_t)  # calls the graphing function and graphs
            break
        elif iteration > 3600:
            print(f"Too many iteration: {iteration}")
            print(f"partial a: {g_k[0]} and partial b:{g_k[1]} and partial c:{g_k[2]}")
            print(f"Parameter a is: {x_next[0]}")
            print(f"Parameter b is: {x_next[1]}")
            print(f"Parameter c is: {x_next[2]}")
            break

        x_k = x_next

main ()