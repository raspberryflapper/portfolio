import matplotlib.pyplot as plt
import numpy as np

def main():
    a=-5
    b=5

    #41 equally spaced x-values and corresponding y-values
    xp = np.linspace(a,b,41)
    yp = f_x(xp)
    print(yp)


    k=0
    while k<=20:
        y_n=0
        xval = float(input("x-value?"))
        for i in range(41):
            p = 1
            for j in range(41):
                if i != j:
                    p = p * (xval - xp[j])/(xp[i] - xp[j])
            y_n = y_n + p * yp[i]
        print(y_n)
        k+=1

def f_x(x):
    return 1/(x*x+1)

main ()