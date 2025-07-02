'''
This code does the following:

1) generate MC interest rate paths for all t in T under Vasicek
2) for each path, at time T1, compute P(T1,T2) - bond price at T1 till T2 (bond maturity)
3) compute option payoff:  max(P(T1,T2)-K,0) path - wise
4) discount the payoff with the path-wise discounting factor (i.e., np.exp(-np.sum(rate_n)*dt))
and find the expected value V_0 - which is your initial value for the European Call option for the bond.

06.16.2025
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# PARAMETERS:

a = 0.3#mean reversion speed
b = 0.05#long term mean
sigma = 0.02
r0 = 0.03#initial interest
T1 = 0.5#option maturity
T2 = 1.0#bond maturity
K = 0.95#strike price for the option
MC_paths = 10000
N = 252#steps, days
dt = (T2 / N)

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
rates_array[:,0] = r0
x
# CALCULATE INTEREST RATES VIA MC:
for path in range(MC_paths):
    for time in range(1,N):
        rates_array[path,time] = np.round((rates_array[path,time-1] + a*(b - rates_array[path,time-1])*dt + (sigma*dW[path,time])),4)
#print("Rates_array at each t: ",rates_array)

def BFunction():
    fn_value = 0
    mat_time = T2 - T1
    fn_value = (1 - np.exp(-a * mat_time)) / a
    return fn_value
b_value_array = BFunction()
#print("b: ",b_value_array)

def AFunction():
    fn_value = 0
    mat_time = T2 - T1
    first_term = (b_value_array - mat_time)
    second_term = ((a**2)*b - (1/2)*(sigma**2)) / a**2
    fn_value = np.exp((first_term * second_term) - ((b_value_array**2 * sigma**2) / (4*a)))
    return fn_value
a_value_array = AFunction()
#print("a: ",a_value_array)

def findT1Rate():
    rate = np.zeros(MC_paths)
    for path in range(MC_paths):
        for n in range(N):
            step = dt * n
            location = 0 + step
            if np.abs(location - T1)<0.00001:
                rate[path] = rates_array[path,n]
                break
    return rate,n
T1_rate,T1_col_number = findT1Rate()
#print(T1_rate)

def bondPriceT1():
    bond = np.zeros_like(T1_rate)
    for i in range(MC_paths):
        bond[i] = a_value_array * np.exp(-b_value_array * T1_rate[i])
    return bond
bond_price_array = bondPriceT1() #1D array of size MC_paths
#print("bond price P(T1,T2) for each paths: ", bond_price_array)

payoffs_T1 = np.zeros_like(bond_price_array)
# max payoff for each paths based on bond price at T1
payoffs_T1 = np.maximum(bond_price_array-K,0) #np.maximum for element wise

discount_factor = np.zeros_like(bond_price_array)
#discount factor for each MC paths
for path in range(MC_paths):
    row = rates_array[path,:]
    discount_factor[path] = np.exp(-np.sum(row[:T1_col_number+1]) * dt) #T1_col_number+1 to include that indexed value

option_initial = np.sum(payoffs_T1 * discount_factor) / MC_paths
print("V_0: ",option_initial)