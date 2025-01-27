def main ():
    guess0=float(input("Your initial guess?"))#x_0
    guess1=float(input("Your second initial guess?"))#x_1
    if 0<=guess0<=1 and 0<=guess1<=1 and f_x(guess0)*f_x(guess1)<0:#f(x_0)*f(x_1)<0 gurantees that the secant line will pass through the x-axis
        findRoot(guess0,guess1)
    else:
        print("Your initial guesses were not in the interval or your secant line does not pass through the x-axis")

def f_x(xvalue):
    import math
    f_x = 2*xvalue*(1-xvalue*xvalue+xvalue)*math.log(xvalue)-(xvalue*xvalue)+1
    return f_x
def findRoot(x0,x1):
    import math
    nPrev = x0
    nCurr  =x1
    nextIter = nCurr-((nCurr-nPrev)/(f_x(nCurr)-f_x(nPrev)))*f_x(nCurr)
    while abs(nextIter-nCurr)>.00001:
        nPrev = nCurr
        nCurr = nextIter
        nextIter = nCurr-((nCurr-nPrev)/(f_x(nCurr)-f_x(nPrev)))*f_x(nCurr)
        #print(nextIter)
        print(f_x(nextIter))
    print(f"Your root is: {nextIter}")

main ()