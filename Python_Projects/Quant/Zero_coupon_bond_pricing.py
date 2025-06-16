'''
We use Vasicek interest rate model to model zero-coupon bond prices.
We use MC to simulate interest rates based on the Vasicek framework.
Currently, as we get closer to the maturity date, the zero-coupon bond
prices all converge to 1.0, the par-value. However, I'm thinking how 
the price decreases to 1.0 is not a correct behavior. Further investigation
is needed.

06.14.2025
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

a = 0.1 #speed of mean reversion
b = 0.06 #long term of mean level
T = 5
r0 = 0.034 #initial interest rate
sigma = 0.2 #volatility 
N = 252
dt = T/N
MC_paths = 100

# fix seed for reproducibility:
seed = 66
rng = np.random.default_rng(seed) #now, use "rng.standard_normal", instead of "np.random.randn"
###REPRODUCIBILITY TOGGLE:
# for fixed seed, uncomment below and comment out the line after:
#dW = rng.standard_normal(size=(MC_paths,N)) * np.sqrt(dt)
# for unfixed seed, uncomment below and comment out above
dW = np.random.randn(MC_paths,N) * np.sqrt(dt)
#print(dW)

rates_array = np.zeros((MC_paths,N))
rates_array[:,0] = r0 #initial rate

# CALCULATE INTEREST RATES VIA MC:
for path in range(MC_paths):
    for time in range(1,N):
        rates_array[path,time] = np.round((rates_array[path,time-1] + a*(b - rates_array[path,time-1])*dt + (sigma*dW[path,time])),4)

#print("Rates_array at each t: ",rates_array)

def plotInterest():
    for y_plot in range(MC_paths):
        plt.plot(np.linspace(0,1,252),rates_array[y_plot])
        plt.xlabel("time")
        plt.ylabel("rates in decimal")
        plt.title("Interest rates")
    return plt.show()
plotInterest()

def BFunction():
    fn_value = np.zeros_like(rates_array)
    for time in range(N):
        mat_time = T - (time*dt)
        fn_value[:,time] = (1-np.exp(-a * mat_time)) / a
    return fn_value
b_value_array = BFunction()
#print("b",b_value_array)

def AFunction():
    fn_value = np.zeros_like(b_value_array)
    for path in range(MC_paths):
        for time in range(N):
            mat_time = T - (time*dt)
            fn_value[path,time] = np.exp((b_value_array[path,time] - mat_time) * (((b*a**2 - ((1/2)*sigma**2)) / a**2) - (b_value_array[path,time]**2 * sigma**2) / (4*a)))
    return fn_value
a_value_array = AFunction()
#print("a",a_value_array)

def bondPricing():
    bond = np.zeros_like(rates_array)
    for path in range(MC_paths):
        for time in range(N):
            bond[path,time] = a_value_array[path,time] * np.exp(-b_value_array[path,time] * rates_array[path,time])
    return bond
bond_price_array = bondPricing()
print(bond_price_array)

final_price_avg = 0
final_price_avg = np.sum(bond_price_array[:,-1]) / MC_paths
print("avg is: ",final_price_avg)

def plotBondPrices():
    for y_plot in range(MC_paths):
        plt.plot(np.linspace(0,1,252),bond_price_array[y_plot])
        plt.xlabel("time")
        plt.ylabel("bond prices")
        plt.title("Bond prices")
    return plt.show()

plotBondPrices()