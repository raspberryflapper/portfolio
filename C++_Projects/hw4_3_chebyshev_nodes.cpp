//
//  hw4_3_chebyshev_nodes.cpp
//  hello.cpp
//
//  Created by SeHwan Kim on 10/28/22.
//

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;
//THIS PROGRAM USES CHEVYSHEV NODES AND LAGRANGE INTERPOLATION
int main ()
{
    float userN;
    float product=1;
    float x[50], y[50],coeffDenom[50];
    int a=-5;
    int b=5;
    const double pi=22.0/7;
    
    cout << "Select your desired nodes (degree n - you have 5, 11,21, and 41 to choose from):"<<endl;
    cin >> userN;
    if (userN==5 || userN==11 || userN==21 || userN==41)
    {
        for (int i=0; i<=userN; i++)
        {
            x[i] = (1/2)*(a+b)+(1/2)*(b-a)*cos(((2*i-1)*pi)/2*userN);
            y[i] = 1/((x[i]*x[i])+1);//values are stored in xi and yi arrays
            cout <<x[i]<<endl;
            cout <<y[i]<<endl;//checks nodes and image values
            /*for (int j=0; j<=userN; j++)
             {
                 if (i != j)
                 {
                    product = (x[i]-x[j])*product;
                    cout<<product<<endl;
                 }
             }coeffDenom[i]=product; //stores denominator into arrays*/
        }
            /*cout << "Your polynmial of degree "<<userN<<" is:"<<endl;
             for (int i=0; i<=userN; i++)
             {
                 cout << y[i]<<"*";
                    for (int j=0; j<=userN; j++)
                        {
                            if (i != j)
                            {
                                cout << "(x-"<<x[j]<<")";
                            }
                        }cout << " all over " <<coeffDenom[i]<<endl;
             }*/
    }else {cout <<"You entered the wrong number of n. Exiting the program"<<endl;}
    
    return 0;
}
