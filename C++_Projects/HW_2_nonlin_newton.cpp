//
//  HW_2_nonlin_newton.cpp
//  c++projects
//
//  Created by SeHwan Kim on 9/21/22.
//

//THIS IS NEWTON'S METHOD FOR F(X) = 1.5872*10^-5(.000003968-7.936*10^-6*x)+.05641728064(-.060402845+.00352608004*x)^3

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
    if (derivCheck==0)
        cout <<"Your initial guess resulted f'("<<initialGuess<<") to be zero. Run the program again and Pick another initial guess"<<endl;
    else
    {
        findRoot(initialGuess);
    }
    return 0;
}

//F(X) = 1.5872*10^-5(.000003968-7.936*10^-6*x)+.05641728064(-.060402845+.00352608004*x)^3
double calcFunc (double xSubk)
{
    return .000015872*(.000003968-.000007936*xSubk)+.05641728064*(-.060402845+.00352608004*xSubk)*(-.060402845+.00352608004*xSubk)*(-.060402845+.00352608004*xSubk);
}
//F'(X) = 1.25960192*10^-10+5.96795542*10^-4*(-.060402845+.00352608004*x)^2
double calcDeriv (double xSubk)
{
    return .000000000125960192+.000596795542*(-.060402845+.00352608004*xSubk)*(-.060402845+.00352608004*xSubk);
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
