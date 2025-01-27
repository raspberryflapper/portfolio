def main ():
    iniGuess=float(input("Your initial guess?"))#x_0
    if 0<=iniGuess<=1 and f_prime(iniGuess)!=0:
        findRoot(iniGuess)
    else:
        print("Your initial guess was not in the interval or it resulted f prime to be zero.")

def f_x(xvalue):
    import math
    f_x = 2*xvalue*(1-xvalue*xvalue+xvalue)*math.log(xvalue)-(xvalue*xvalue)+1
    return f_x
def f_prime(xvalue):
    import math
    f_prime = 2*xvalue-2*xvalue*xvalue*xvalue+2*xvalue*xvalue*xvalue*(1/xvalue)+math.log(xvalue)*(2-6*xvalue*xvalue+4*xvalue)-2*xvalue
    return f_prime
def findRoot(xvalue):
    import math
    nextIter = xvalue-f_x(xvalue)/f_prime(xvalue)
    while abs(nextIter-xvalue)>.00001:
        xvalue = nextIter
        nextIter = xvalue-f_x(xvalue)/f_prime(xvalue)
        #print(nextIter)
        #print(f_x(nextIter))
    print(f"Your root is: {nextIter}")

main ()