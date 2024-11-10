'''
You can analyze multiple parameter values found with this code.
E.g. use it with different runs and results from multiple gradient descent
and analyze multiple RSS.
'''
import numpy as np
import math

def main():
    #below are the collected data points from the lab
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
    w interpolation:
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
    no interpolation:
    [(0,0.8),
    (180,0.81),
    (360,0.82),
    (540,0.41),
    (720,0.22)]
    '''

    #store x values (time) of the data into array t
    t = np.array([data[0] for data in data_1a])
    #store y values (number of cells) of the data into array n
    n = np.array([data[1] for data in data_1a])

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

    #parameter values found with the gradient descent
    parameter_results = [
                (-1.290991),
                (54.99227),
                (6.180892),
                (0.9939386)]
    #each rows are a,b,c,and d results from each run
    #so, the first column represent a,b,c,d values from run #1

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
        #and assign them to a,b,c,d respectively.
        a,b,c,d = run[i]
        margin = 0
        for m in range(len(norm_t)):
            margin += ((a * np.exp(-b * np.exp(-c * norm_t[m]))+d)-(norm_n[m]))**2 #margin = residual sum of squares
        print(f"run #{i}", margin)
    
    
main()