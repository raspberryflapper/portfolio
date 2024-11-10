'''
You can analyze multiple parameter values found with this code.
E.g. use it with different runs and results from multiple gradient descent
and analyze multiple RSS.
'''
import numpy as np
import math

# y_1 function -the modeling function for 1a, 1b data
def y_1(a, b, c, t):
    return a / ( 1+np.exp(b*(t-c)) )

# y_2 function - the modeling function for 2a,2b data
def y_2(a, b, c, t):
    return a*(t**b)*np.exp(-c*t)

def main():
    #below are the collected data points from the lab
    #go to the bottom for data list
    data_list = [
                (0.01,0.1),
                (15,0.8),
                (30,0.41),
                (45,0.2),
                (60,0.1),
                (90,0.2)]

    #store x values (time) of the data into array t
    t = np.array([data[0] for data in data_list])
    #store y values (number of cells) of the data into array n
    n = np.array([data[1] for data in data_list])

    #normalization of the data
    min_t = float(min(t))
    max_t = float(max(t))
    norm_t = []
    for i in range(len(t)):
        value = float((t[i] - min_t) / (max_t - min_t))
        norm_t.append(value)
    min_n = float(min(n))
    max_n = float(max(n))
    norm_n =[]
    for i in range(len(n)):
        value = float((n[i] - min_n) / (max_n - min_n))
        norm_n.append(value)

    #parameter values found with steepest gradient descent
    parameter_results = [
                (2.9286351),
                (0.27616997),
                (4.55679943)]
    #each rows are a,b, and c results from each run
    #so, the first column represent a,b,and c values from run #1

    #store parameter values into arrays 
    run = [] #list to put our arrays

    #use below for multiple columns (i.e. results of parameters)
    '''
    for i in range(1):
        run.append(np.array([param[i] for param in parameter_results]))
    #Print the results to verify
    for i in range(1):
        print(f"run[{i}]:", run[i])
    '''
    #use for single result/column:
    run.append(np.array(parameter_results))
    #print to verify
    for i in range(len(run)):
        print(f"run[{i}]:", run[i])
    
    
    for i in range(1):
        #"unpack" each elements, the parameter values, from the ith array
        #and assign them to a,b, and c respectively.
        a,b,c = run[i]
        margin = 0
        for m in range(len(norm_t)):
            #for y_1:1a and 1b
            #margin += ((y_1(a,b,c,norm_t[m]))-(norm_n[m]))**2 #margin = residual sum of squares
            #for y_2:2a and 2b
            margin += ((y_2(a,b,c,norm_t[m]))-(norm_n[m]))**2
        print(f"run #{i}", margin)
    
    
main()

'''
1a with interpolation:
[
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

1a wo int:
[
    (0,0.8),
    (180,0.81),
    (360,0.82),
    (540,0.41),
    (720,0.22)]

1b wo int:
[
    (0.01,0.8),
    (200,0.8),
    (400,0.8),
    (550,0.4),
    (720,0.2)]

2a wo int:
[
    (0.01,0.1),
    (15,0.8),
    (30,0.41),
    (45,0.2),
    (60,0.1),
    (90,0.2)]

2b wo int
[
    (0.01,0.01),
    (20,0.7),
    (30,0.6),
    (45,0.4),
    (60,0.3),
    (90,0.2)]
'''