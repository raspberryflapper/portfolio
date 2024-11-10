import numpy as np
import math
import matplotlib.pyplot as plt
import timeit

# graphs my results
def graph_data(x, y, x_2, y_2,user_choice):
    plt.figure()
    # scatter is the lab data
    plt.scatter(x,y, color = 'red')
    # plot is the data fit with steepest GD
    plt.plot(x_2,y_2, color = 'purple')
    plt.xlabel('Time in minutes - normalized')
    plt.ylabel('p_NFKB -noramlized')
    if user_choice =='1a':
        plt.title('Over expression of HER2. Block HER2 (in vivo)-1a')
    elif user_choice =='1b':
        plt.title('Over expression of HER2. Block HER2 and IL1 1b (in silico)')
    elif user_choice =='2a':
        plt.title('Over expression of HER2. Reach equilibrium. Add IL1 (in vivo) - 2a')
    elif user_choice =='2b':
        plt.title('Over expression of HER2. Reach equilibrium. Block HER2. Reach equilibrium. Add IL1 (in vivo)-2b')
    return plt.show()

def validate_interval(df, x0, x1):
    #print(df(x0), df(x1))
    return df(x0) * df(x1) < 0

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

# y_1 function -the modeling function for 1a, 1b data
def y_1(a, b, c, t):
    return a / ( 1+np.exp(b*(t-c)) )

# y_2 function - the modeling function for 2a,2b data
def y_2(a, b, c, t):
    return a*((t+0.0000001)**b)*np.exp(-c*t)

# gradient of phi - returns 3x1 vector with partials calculated
def grad(x_k, norm_t, norm_n, user_select):
    # calls scalar value of x_k vector's elements
    a = x_k[0].item()
    b = x_k[1].item()
    c = x_k[2].item()

    partial_a = 0
    partial_b = 0
    partial_c = 0

    if user_select in ['1a', '1b']:
        # for y_1
        for i in range(len(norm_t)):
            # this is the common factor in all three partials
            y_1_minus_n = (y_1(a,b,c,norm_t[i]) - norm_n[i])
            # summations
            partial_a += (1 / ( 1+np.exp(b*(norm_t[i]-c)) )) * (y_1_minus_n)
            partial_b += ( ( -a * (np.exp(b*(norm_t[i]-c))*(norm_t[i]-c)) ) / ( 1+np.exp(b*(norm_t[i]-c)) )**2 ) * (y_1_minus_n)
            partial_c += ( (-a * (np.exp(b*(norm_t[i]-c))*-b))  / ( 1+np.exp(b*(norm_t[i]-c)) )**2 ) * (y_1_minus_n)
    elif user_select in ['2a', '2b']:
        # for y_2
        for i in range(len(norm_t)):
            # common factor "y_2(t) - n(t)"
            y_2_minus_n = (y_2(a,b,c,norm_t[i]) - norm_n[i])
            # summation
            partial_a += (y_2_minus_n) * (norm_t[i] * np.exp(-c*norm_t[i]))
            partial_b += (y_2_minus_n) * (a*np.exp(-c*norm_t[i])*((norm_t[i]+0.000001)**b)*np.log(norm_t[i]+0.000001))
            partial_c += (y_2_minus_n) * (-a*(norm_t[i]+0.000001)**(b+1)*np.exp(-c*norm_t[i]))
    else:
        raise ValueError
    
    # create array of column vectors, 3x1, initialized with elements=0 to store partial values 
    grad_result = np.zeros((3, 1))
    # store partial a,b,c values into array g, which is a 3x1 column vector
    grad_result[0] = partial_a
    grad_result[1] = partial_b
    grad_result[2] = partial_c
    
    return grad_result
def userSelectData(selection):
    data_list = []
    if selection == '1a':
        data_list = [
                    (0,0.8),
                    (180,0.81),
                    (360,0.82),
                    (540,0.41),
                    (720,0.22)]
        return data_list
    if selection == '1b':
        data_list = [
                    (0.01,0.8),
                    (200,0.8),
                    (400,0.8),
                    (550,0.4),
                    (720,0.2)]
        return data_list
    if selection == '2a':
        data_list = [
                    (0.01,0.1),
                    (15,0.8),
                    (30,0.41),
                    (45,0.2),
                    (60,0.1),
                    (90,0.2)]
        return data_list
    if selection == '2b':
        data_list = [
                    (0.01,0.01),
                    (20,0.7),
                    (30,0.6),
                    (45,0.4),
                    (60,0.3),
                    (90,0.2)]
        return data_list
    else:
        raise ValueError("Invalid input. Try again")

def normalizeT(data_lab):
    # store t values (time) of the data into array t
    t = np.array([data[0] for data in data_lab])
    t_normalized = normalize(t)
    return t_normalized

def normalizeN(data_lab):
     # store y values ('p-NFKB') of the data into array n
    n = np.array([data[1] for data in data_lab])
    n_normalized = normalize(n)
    return n_normalized

def normalize(value_list):
    min_val = float(min(value_list))
    max_val = float(max(value_list))
    normalized_list = []
    for i in range(len(value_list)):
        value = float((value_list[i] - min_val) / (max_val - min_val))
        normalized_list.append(value)
    return normalized_list


def main():
    # user selects which data set to fit
    select_data = input("Select the data set you want: (1a),(1b),(2a) and (2b)")
    # store and return user selected lab data set 
    lab_data = userSelectData(select_data) #lab_data is the list containing lab data
    # normalize the list "lab_data" into normalized lists
    norm_t = normalizeT(lab_data) # stores x values normalized
    norm_n = normalizeN(lab_data) # stores y values normalized

    
    # initialize variables such as a, b, and c
    iteration = 0
    # array to hold a,b, and c values - put in intial a,b, and c values below
    x_k = np.array([1, 1, 1])
    # transpose the array "x_k", 1x3, above to vector 3x1
    x_k = np.reshape(x_k, (3, 1))

    # calculate and solve the parameters using the gradient descent method
    while True:
        iteration += 1
        g_k = grad(x_k, norm_t, norm_n, select_data)

        def d_ls_fun(z):
            dx_k = g_k
            x_k1 = x_k - z * dx_k
            dx_k1 = grad(x_k1, norm_t, norm_n, select_data)
            return np.dot(dx_k.T, dx_k1) # dot the two gradients to approximate the derivative of gamma of step size (i.e. argmin function)

        # find the zeros of the approximation of the optimal step size function
        step_size = bisection(d_ls_fun,-10,10)

        # use the found step size and x_k and g_k to update to x_{k+1}
        x_next = x_k - step_size * g_k
        
        print(f"iteration: {iteration}")
        print(f"partial a: {g_k[0]}, partial b:{g_k[1]}, partial c: {g_k[2]}")

        # termination condition
        if np.linalg.norm(x_next - x_k) / np.linalg.norm(x_k) < 1e-10:
        # if abs(g_k[0])<1e-5 and abs(g_k[1])<1e-5 and abs(g_k[2])<1e-5:
            print(f"After {iteration} number of iterations: ")
            print(f"Parameter a is: {x_next[0]} and partial A is {g_k[0]}")
            print(f"Parameter b is: {x_next[1]} and partial B is {g_k[1]}")
            print(f"Parameter c is: {x_next[2]} and partial C is {g_k[2]}")

            # Use the parameters found for the function f(t). Get and store the output values for the graph
            x = np.linspace(0, 1, 100)
            f_t = []
            if select_data in ['1a', '1b']:
                for t in x:
                    function_t = y_1(x_next[0],x_next[1],x_next[2],t)
                    f_t.append(function_t)
            elif select_data in ['2a', '2b']:
                for t in x:
                    function_t = y_2(x_next[0],x_next[1],x_next[2],t)
                    f_t.append(function_t)
            else:
                raise ValueError 
            graph_data(norm_t, norm_n, x, f_t, select_data)  # calls the graphing function and graphs
            break
        elif iteration > 3600:
            print(f"Too many iteration: {iteration}")
            print(f"partial a: {g_k[0]}, partial b:{g_k[1]}, partial c: {g_k[2]}")
            print(f"Parameter a is: {x_next[0]}")
            print(f"Parameter b is: {x_next[1]}")
            print(f"Parameter c is: {x_next[2]}")
            x = np.linspace(0, 1, 100)
            f_t = []
            if select_data in ['1a', '1b']:
                for t in x:
                    function_t = y_1(x_next[0],x_next[1],x_next[2],t)
                    f_t.append(function_t)
            elif select_data in ['2a', '2b']:
                for t in x:
                    function_t = y_2(x_next[0],x_next[1],x_next[2],t)
                    f_t.append(function_t)
            else:
                raise ValueError 
            graph_data(norm_t, norm_n, x, f_t, select_data)
            break

        x_k = x_next

main ()