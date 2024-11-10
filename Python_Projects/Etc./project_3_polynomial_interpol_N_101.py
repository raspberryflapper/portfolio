import matplotlib.pyplot as plt
import numpy as np

def main():
    a=0
    b=5

    #41 equally spaced x-values and corresponding y-values
    xp = np.linspace(a,b,101)
    """
    for i in range(101):
        print(xp[i])"""
    yp = f_x(xp)
    #print(yp)

    myXs = [0.0]*204
    myXs[0]=a
    myYs = [0.0]*204
    deltaX = abs(b-a)/200
    for i in range(202):
        myXs[i+1]=myXs[i]+deltaX
        #print(myXs[i])
        myYs[i] = f_x(myXs[i])
        #print(myYs[i])

    k=0
    while k<=20:
        y_n=0
        xval = float(input("x-value?"))
        for i in range(101):
            p = 1
            for j in range(101):
                if i != j:
                    p = p * (xval - xp[j])/(xp[i] - xp[j])
            y_n = y_n + p * yp[i]
        print(y_n)
        k+=1
        

def f_x(x):
    return 1/(x*x+1)

main ()