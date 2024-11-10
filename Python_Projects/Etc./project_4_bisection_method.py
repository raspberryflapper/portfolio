import numpy as np
import math

def my_bisection(f, a, b, tol):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception(
         "The scalars a and b do not bound a root")

    # get midpoint
    m = (a + b)/2

    if np.abs(f(m)) < tol:
        # stopping condition, report m as root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        return my_bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        return my_bisection(f, a, m, tol)
def main ():
    f = lambda xvalue: 2*xvalue*(1-xvalue*xvalue+xvalue)*math.log(xvalue)-(xvalue*xvalue)+1
    a = float(input("What is your a?"))
    b = float(input("What is your b?"))
    root = my_bisection (f,a,b,.00001)
    print(root)

main ()