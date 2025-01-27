def f_x(x):
    return 1/(x*x+1)

def main ():
    import matplotlib.pyplot as plt

    a=0
    b=5
    deltaX = abs((b-a)/202)

    x_n = [0.0]*204
    x_n[0]=a
    y_n = [0.0]*204
    for i in range(202):
        x_n[i+1]=x_n[i]+deltaX
        #print(x_n[i])
        y_n[i]=f_x(x_n[i])
        #print(y_n[i])

    n = 101
    deltaKnot = float(abs((b-a)/n))
    t_n = [0.0]*120 #knots
    knot_y_n =[0.0]*120
    t_n[0]=a
    for i in range (n+1):
        t_n[i+1]=t_n[i]+deltaKnot
        #print(t_n[i])
        knot_y_n[i] = f_x(t_n[i])
        #print(knot_y_n[i])

    z_n = [0.0]*102
    z_n[0]=0

    for i in range(0,n):
        z_n[i+1]=-z_n[i]+2*((knot_y_n[i+1]-knot_y_n[i])/(t_n[i+1]-t_n[i]))
        #print(f"{i}th: {z_n[i]}")

    for i in range(50,n):
        print(f"{t_n[i]} to {t_n[i+1]}:")
        for k in range(202):
            if t_n[i]<=x_n[k]<=t_n[i+1]:
                quadEval = ((z_n[i+1]-z_n[i])/2*(t_n[i+1]-t_n[i]))*(x_n[k]-t_n[i])**2+z_n[i]*(x_n[k]-t_n[i])+knot_y_n[i]
                print(quadEval)

main ()