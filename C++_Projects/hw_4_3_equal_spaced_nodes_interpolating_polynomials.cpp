//
//  hw_4_3_interpolating_polynomials.cpp
//  hello.cpp
//
//  Created by SeHwan Kim on 10/27/22.
//


#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;
//THIS PROGRAM USES LAGRANGE INTERPOLATION
int main ()
{
    float userN;
    float deltaX, deltaXP, xp, yp, p;
    float x[2000], y[2000],xP[50], yF[50], yP[2000];
    float h=1000;
    
    deltaX=10/h;
    x[0]=-5;
    y[0]=1/((x[0]*x[0])+1);
    for (int i=0; i<=h; i++)
    {
        x[i+1] = x[i]+deltaX;
        y[i+1] = 1/((x[i+1]*x[i+1])+1);
    }
    cout << "Select your desired nodes (degree n - you have 5, 11,21, and 41 to choose from):"<<endl;
    cin >> userN;
    if (userN==5 || userN==11 || userN==21 || userN==41)
    {
        deltaXP = 10/userN;
        xP[0]=-5;
        yF[0] = 1/26;
        for (int i=0; i<=userN; i++)
        {
            xP[i+1] = xP[i]+deltaXP;
            yF[i+1] = 1/((xP[i+1]*xP[i+1])+1);
        }
        for (int i = 0; i<=h; i++)
                {
                    xp=x[i];
                    yp=0;
                    for (int j=0; j<=userN; j++)
                    {
                        p = 1;
                        for (int k=0; k<=userN; k++)
                        {
                            if (xP[j] != xP[k])
                            {
                                p = p* (xp - xP[k])/(xP[j] - xP[k]);
                            }
                        }yp = yp + p * yF[j];
                    }yP[i]=yp;
                }
        cout <<"*****The P(x_i) values for P are below*****"<<endl;
        for (int i=0; i<=h; i++)
        {
            cout <<yP[i]<<endl;
        }
    }else {cout <<"You entered the wrong number of n. Exiting the program"<<endl;}

    return 0;
}
