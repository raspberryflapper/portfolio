import numpy as np
import math

def main():
    #below are the collected data points from the lab
    data_1a = [
                (0,0.8),
                (180,0.81),
                (360,0.82),
                (540,0.41),
                (720,0.22)]

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
    a= -1.169367341715096
    b= 51.41919201289323
    c= 5.954712331886254
    d= 1.0045788523766295

    margin = 0

    for i in range(len(norm_t)):
        margin += ((a * np.exp(-b * np.exp(-c * norm_t[i]))+d)-(norm_n[i]))**2 #margin = residual sum of squares
    print(f"RSS: ", margin)
            
    
main()