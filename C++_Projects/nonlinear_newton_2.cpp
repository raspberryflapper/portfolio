//
//  linear_newton_2.cpp
//  hello.cpp
//
//  Created by SeHwan Kim on 9/13/22.
//

//THIS IS NEWTON'S METHOD FOR F(X) = 2X(1-X^2+X)LN(X)=X^2-1

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#define EPSILON .000001

using namespace std;

double calcFunc (double xSubk);
double calcDeriv (double xSubk);
void findRoot (double xSubk);

int main ()
{
    double initialGuess;
    cout <<"Input your initial guess (i.e., x_0): ";
    cin >>initialGuess;
    double derivCheck = calcDeriv(initialGuess); //checks if f prime will be 0 with the intial guess
    if (initialGuess<0 || 1<initialGuess)
        cout <<"Your initial guess is not in the interval. Exiting the program"<<endl;
    else if (derivCheck==0)
        cout <<"Your initial guess resulted f'("<<initialGuess<<") to be zero. Run the program again and Pick another initial guess"<<endl;
    else
    {
        findRoot(initialGuess);
    }
    return 0;
}

double calcFunc (double xSubk)
{
    return 2*xSubk*(1-(xSubk*xSubk)+xSubk)*log(xSubk)-(xSubk*xSubk)+1;
}
double calcDeriv (double xSubk)
{
    return (2*xSubk-2*xSubk*xSubk*xSubk+2*xSubk*xSubk)*(1/xSubk)+log(xSubk)*(2-6*xSubk*xSubk+4*xSubk)-2*xSubk;
}
void findRoot (double xSubk)
{
    double nextIter = xSubk - calcFunc(xSubk) / calcDeriv(xSubk);
    while (abs(nextIter-xSubk)> EPSILON)
    {
        xSubk = nextIter;
        nextIter = xSubk - calcFunc(xSubk) / calcDeriv(xSubk);
        cout <<nextIter<<endl;
    }
    cout << "Your root is: "<<nextIter<<endl;
}
