//
//  linear.cpp
//  hello.cpp
//
//  Created by SeHwan Kim on 9/13/22.
//

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#define EPSILON .000001
//for the newton method, the user must pick x_0(i.e., f(x_0) then is evaluated and the derivative as well)
using namespace std;

double calcFunc (double xSubk);
double calcDeriv (double xSubk);
void findRoot (double xSubk);

int main ()
{
    double initialGuess;
    cout <<"Input your initial guess (i.e., x_0): ";
    cin >>initialGuess;
    int derivCheck = calcDeriv(initialGuess); //checks if f prime will be 0 with the intial guess
    if (initialGuess<-3 || 3<initialGuess)
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
    return xSubk * xSubk * xSubk - 5*xSubk+3;
}
double calcDeriv (double xSubk)
{
    return 3* xSubk * xSubk - 5;
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
