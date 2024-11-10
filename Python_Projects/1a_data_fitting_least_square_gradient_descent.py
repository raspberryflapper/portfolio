"""
READ:
SeHwan Kim
11.10.2024


This code is for data fitting certain sets of data. The data is fitted in the least square sense
with Gradient Descent as the optimizer. In this example,
based on the lab data provided, tumor growth effected by HER2 cell signaling, we suspect a certain
mathematical function would fit the data. The data is normalized first
and Gradient Descent is used to find the unknown parameters to
minimize the cost function (i.e.e the least square error).

This project was done in Summer of 2024 with my
undergraduate collaborators for a research project.

Complex terms (e.g. multiple "np.exp()" in a single term) are
delegated to variables rather than typed out for readability and to minimize
any compiler/calculation issues (used parenthesis heavilty to separate variables and
calculations as much as possible) in the code.
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import timeit

def graphData(x,y,x_2,y_2):
    plt.figure()
    plt.scatter(x,y, color = 'purple')
    plt.plot(x_2,y_2, color = 'green')
    plt.xlabel('Time in minutes - normalized')
    plt.ylabel('p_NFKB -noramlized')
    plt.title('Over expression of HER2. Reach equilibrium. Block HER2 (in vivo) 1a')
    return plt.show()

def main():
    #estimated data points according to the graph (slide #14)
    #5 data points in 1a
    data_1a = [
                (0, 0.8),
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
    1a with interpolation:
    [   (0, 0.8),
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
    1a wo interpolation
    [
        (0,0.8),
        (180,0.81),
        (360,0.82),
        (540,0.41),
        (720,0.22)]
    '''
    #store x values (time in minutes) into array t
    t = np.array([data[0] for data in data_1a])
    #t = t/72
    #store y values into array y
    y = np.array([data[1] for data in data_1a])

    #normalize the x's
    min_t = float(min(t))
    max_t = float(max(t))
    norm_t = []
    for i in range(len(t)):
        value = float((t[i] - min_t) / (max_t - min_t))
        norm_t.append(value)

    #normalize the y's
    norm_y = []
    min_y = float(min(y))
    max_y = float(max(y))
    for i in range(len(y)):
        new_y = (y[i]-min_y)/(max_y-min_y)
        norm_y.append(new_y)
        #print(norm_y[i])
    

    #ask user for initial parameter guesses and step size
    a = float(input("Initial value for a: "))
    b = float(input("Initial value for b: "))
    c = float(input("Initial value for c: "))
    d = float(input("Initial value for d: "))
    step_size = float(input("Step size: "))
    #store user initial guess for running the program again
    initial_a = a
    initial_b = b
    initial_c = c
    initial_d = d
    #previous parameter values for the gradient descent algorithm
    prev_a = a
    prev_b = b
    prev_c = c
    prev_d = d
    #initialize iteration number
    iteration = 0

    #calculate using gradient descent
    while True:
        iteration +=1
        #initialize/reset partial_a,b,c values
        partial_a = 0
        partial_b = 0
        partial_c = 0
        partial_d = 0


        """
        #previous messy 4 different loops calculations
        for i in range(len(t)):
            partial_a += (a * np.exp(-2*b * np.exp(-c * norm_t[i])) - (norm_y[i] * np.exp(-b * np.exp(-c * norm_t[i]))) + (d * np.exp(-b * np.exp(-c * norm_t[i]))))
        for i in range(len(t)):
            partial_b += (-(a**2) * np.exp(-c*norm_t[i]-2*b*np.exp(-c*norm_t[i])) + (norm_y[i] * a * np.exp(-c*norm_t[i] - b*np.exp(-c*norm_t[i]))) - (a*d* np.exp(-c*norm_t[i] - b*np.exp(-c*norm_t[i]))))
        for i in range(len(t)):
            partial_c += ((a**2) *b*norm_t[i]*np.exp(-c*norm_t[i]-2*b*np.exp(-c*norm_t[i])) - (norm_y[i]*a*b*norm_t[i]*np.exp(-c*norm_t[i] - b*np.exp(-c*norm_t[i]))) + (norm_t[i]*a*b*d* np.exp(-c*norm_t[i] - b * np.exp(-c*norm_t[i]))))
        for i in range(len(t)):
            partial_d += (a*np.exp(-b*np.exp(-c*norm_t[i])) + d - norm_y[i])
        """
        
        #now, consolidated into one loop and clean it up:
        for i in range(len(norm_t)):
            #this is e^(-b*e^(-c*t))
            exp_term_no_coeff = np.exp(-b*np.exp(-c*norm_t[i]))
            #3 terms in side the sum
            inside_terms = (a*np.exp(-b*np.exp(-c*norm_t[i]))+d-norm_y[i])
            #e^((-c*t)-(b*e^(-c*t)))
            exp_term_minus_ct = np.exp((-c*norm_t[i])-b*np.exp(-c*norm_t[i]))
            #summations
            partial_a += (inside_terms)*(exp_term_no_coeff)
            partial_b += (inside_terms)*(-a)*(exp_term_minus_ct)
            partial_c += (inside_terms)*(exp_term_minus_ct)*(a*b*norm_t[i])
            partial_d += (inside_terms)
        

        #gradient descent algorithm:
        #update the parameters a,b, and c
        a = prev_a  - (step_size * partial_a)
        b = prev_b  - (step_size * partial_b)
        c = prev_c  - (step_size * partial_c)
        d = prev_d  - (step_size * partial_d)

        #print (f"{iteration}th iteration with")
        #print (f"a:{a}, b:{b}, c:{c}, d:{d}")

        if abs(partial_a)<0.0001 and abs(partial_b)<0.0001 and abs(partial_c)<0.0001 and abs(partial_d)<0.0001:
            print (f"After {iteration}th iterations: ")
            print (f"Parameter a is: {a} and partial A is {partial_a}")
            print (f"Parameter b is: {b} and partial B is {partial_b}")
            print (f"Parameter c is: {c} and partial C is {partial_c}")
            print (f"Parameter d is: {d} and partial C is {partial_d}")
            #Use the parameters found here for the function f(norm_t). Get and store the output values for the graph
            x = np.linspace(0,1,100)
            f_t = []
            for i in range(len(x)):
                function_t = a * np.exp(-b * np.exp(-c * x[i]))+d
                f_t.append(function_t)
            graphData(norm_t,norm_y,x,f_t)#calls the graphing function and graphs
            break
        
        elif iteration >30000000:
            print("Too many iterations")
            print(initial_a,initial_b,initial_c,initial_d)
            print("initial a, b, c, and d above.")
            print(step_size)
            print("step size above.")
            break

        prev_a = a
        prev_b = b
        prev_c = c
        prev_d = d

if __name__ == "__main__":
    # Measure execution time using timeit
    execution_time = timeit.timeit("main()", globals=globals(), number=1)
    
    # Print execution time
    print(f"Execution time: {execution_time:.4f} seconds")