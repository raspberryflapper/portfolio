def f_x(x):
    return 1/(x*x+1)

def main ():
    import matplotlib.pyplot as plt

    a=0
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

    n = 101
    deltaKnot = float(abs((b-a)/n))
    t_n = [0.0]*103 #knots
    knot_y_n =[0.0]*103
    t_n[0]=a
    for i in range (n+1):
        t_n[i+1]=t_n[i]+deltaKnot
        #print(t_n[i])
        knot_y_n[i] = f_x(t_n[i])
        #print(knot_y_n[i])

    h_n = [0.0]*102
    b_n = [0.0]*102

    for i in range (n+1):
        h_n[i]=t_n[i+1]-t_n[i]
        #print(h_n[i])
        b_n[i]=1/h_n[i]*(knot_y_n[i+1]-knot_y_n[i])

    u_n=[0.0]*102
    v_n=[0.0]*102
    u_n[1] = 2*(h_n[0]+h_n[1])
    v_n[1] = 6*(b_n[1]-b_n[0])

    for i in range(2,n):
        u_n[i] = 2*(h_n[i]+h_n[i-1])-(h_n[i-1]*h_n[i-1])/u_n[i-1]
        v_n[i] = 6*(b_n[i]-b_n[i-1])-(h_n[i-1]*v_n[i-1])/u_n[i-1]

    z_n = [0.0]*102
    z_n[0]=0

    for i in range(1,n):
        z_n[i]=(v_n[i]-h_n[i]*z_n[i+1])/u_n[i]
        #print(f"{i}th: {z_n[i]}")

    for i in range(n+1):
        #print(f"{t_n[i]} to {t_n[i+1]}:")
        for k in range(200):
            if t_n[i]<=x_n[k]<=t_n[i+1]:
                h=t_n[i+1]-t_n[i]
                tmp = (z_n[i]/2)+(x_n[k]-t_n[i])*(z_n[i+1]-z_n[i])/(6*h)
                tmp = -(h/6)*(z_n[i+1]+2*z_n[i])+(knot_y_n[i+1]-knot_y_n[i])/h+(x_n[k]-t_n[i])*tmp
                splineEval = knot_y_n[i]+(x_n[k]-t_n[i])*tmp
                print(splineEval)

main ()