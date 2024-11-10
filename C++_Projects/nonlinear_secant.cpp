//
//  linear_secant.cpp
//  hello.cpp
//
//  Created by SeHwan Kim on 9/13/22.
//

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#define EPSILON .000001

using namespace std;

double calcFunc0 (double x0);
double calcFunc1 (double x1);
void findRoot (double x0, double x1);

//for the secant method, the user must pick x_0 and x_1 (i.e., two points must be picked to draw a chord between the two functional values
int main ()
{
    double initialGuess0, initialGuess1;
    cout <<"Input your initial guess (i.e., x_0): ";
    cin >>initialGuess0;
    cout <<"Input your second initial guess (i.e., x_1): ";
    cin >>initialGuess1;
    
    //checks f(a)f(b)<0
    double checkVal;
    checkVal = calcFunc0(initialGuess0) * calcFunc1(initialGuess1);
    if (initialGuess0<-3 || 3<initialGuess0 || initialGuess1<-3 ||3<initialGuess1)
        cout <<"Your initial guesses were not in the interval. Exiting the program"<<endl;
    else if (checkVal>0)//checks for f(a)f(b)<0
        cout <<"Your initial guesses were not sufficient and the secant line does not intersect the x-axis. Run the program again and Pick another initial guess"<<endl;
    else
    {
        findRoot(initialGuess0, initialGuess1);
    }
    return 0;
}

double calcFunc0 (double x0)
{
    return x0 * x0 * x0 - 5*x0+3;
}
double calcFunc1 (double x1)
{
    return x1 * x1 * x1 - 5*x1+3;
}
void findRoot (double x0, double x1)
{
    double nextIter = x1-calcFunc1(x1)*((x1-x0)/(calcFunc1(x1)-calcFunc0(x0)));//e.g. x_2 is defined and calculated here
    while (abs(nextIter-x1)> EPSILON)
    {
        x0=x1;//assigns x0=x1
        x1 = nextIter;//assigns x1=x_2
        nextIter = x1-calcFunc1(x1)*((x1-x0)/(calcFunc1(x1)-calcFunc0(x0)));//e.g. calculates x_3 here.x1 here is the previous 'nextIter' term = x_2. essentially, every term gets bumped up to the next
        cout <<nextIter<<endl;
    }
    cout << "Your root is: "<<nextIter<<endl;
}
