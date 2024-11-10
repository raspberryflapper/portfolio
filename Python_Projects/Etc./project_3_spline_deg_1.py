def f_x(x):
    return 1/(x*x+1)

def main ():
    import matplotlib.pyplot as plt

    a=-5
    b=5
    deltaX = abs((b-a)/200)

    x_n = [0.0]*204
    x_n[0]=a
    y_n = [0.0]*204
    for i in range(200):
        x_n[i+1]=x_n[i]+deltaX
        #print(x_n[i])
        y_n[i]=f_x(x_n[i])
        #print(y_n[i])

    n = 41
    deltaKnot = float(abs((b-a)/n))
    k_n = [0.0]*50 #knots
    k_n[0]=a
    for i in range (n+1):
        k_n[i+1]=k_n[i]+deltaKnot
        #print(k_n[i])

    for i in range(21,n+1):
        print(f"{k_n[i-1]} to {k_n[i]}:")
        for xval in range(200):
            if k_n[i-1]<=x_n[xval]<=k_n[i]:
                s=f_x(k_n[i-1])*((k_n[i]-x_n[xval])/(k_n[i]-k_n[i-1]))+f_x(k_n[i])*((x_n[xval]-k_n[i-1])/(k_n[i]-k_n[i-1]))
                print(s)
main ()