'''
This code executes and calculates Monte Carlo pricing of a European Call option
on a zero coupon bond under Hull-White model. We calibrate the Theta(t) from the
forward rate and it is incorporated in the Monte Carlo Vasicek paths for the 
interest rates. Then, it calculates MC bond pricing, and the expected payoffs
with discounting.
06.30.2025
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# PARAMETERS:
a = 0.1#mean reversion speed
sigma = 0.01
r0 = 0.03#initial interest rate - assume f(0,t) = r0
K = 0.8 #strike price
T_1 = 0.5#call option maturity
T_2 = 1.0#bond maturity
N = 100
dt = T_2 / N
MC_paths = 10
seed = 70
rng = np.random.default_rng(seed)
dW = rng.standard_normal(size=(MC_paths,N)) * np.sqrt(dt) #for fixed seed
#dW =  np.random.randn(MC_paths,N) * np.sqrt(dt)

def calcPFn(t):
    array = np.zeros(N)
    t_vals = np.linspace(0,t,N)
    for time in range(N):
        array[time] = np.exp(-r0 * t_vals[time])
    return array
p_array_T2 = calcPFn(T_2)

def forwardRate(array):
    forward = -1 * np.gradient(np.log(array),dt)
    return forward
forward_array_T2 = forwardRate(p_array_T2)

def driftTermTheta():
    theta = np.zeros((N))
    theta = np.gradient(forward_array_T2,dt) + (a * forward_array_T2) + (sigma**2 / (2*a))*(1-np.exp(-2*a*np.linspace(0,T_2,N)))
    return theta
theta_array = driftTermTheta()
#print("Theta(t_i): ",theta_array)

# CALCULATE INTEREST RATES VIA MC:
rates_MC_array = np.zeros((MC_paths,N))
rates_MC_array[:,0] = r0
#MC generate interest rates for every t
for paths in range(MC_paths):
    for time in range(1,N):
        rates_MC_array[paths,time] = np.round(rates_MC_array[paths,time-1]+(theta_array[time-1]-(a * rates_MC_array[paths,time-1]))*dt + (sigma * dW[paths,time]),6)
#print("Rates MC: ", rates_MC_array)

def findT1Rate():
    rate = np.zeros(MC_paths)
    for path in range(MC_paths):
        for n in range(N):
            step = dt * n
            location = 0 + step
            if np.abs(location - T_1)<0.00001:
                rate[path] = rates_MC_array[path,n]
                break
    return rate,n
T1_rate,T1_col_number = findT1Rate()
#print("Rates at T_1 for each paths: ",T1_rate)
#print(T1_col_number)

def BFunction(t):
    fn_value = 0
    mat_time = T_2 - t
    fn_value = (1 - np.exp(-a * mat_time)) / a
    return fn_value
b_value_T1 = BFunction(T_1)
print("B(T_1,T_2): ",b_value_T1)

def AFunction(t):
    fn_value = 0
    first_term = b_value_T1 * forward_array_T2[T1_col_number]
    second_term = ((sigma**2) / (4*a)) * (b_value_T1**2) * (1-np.exp(-2*a*T_1))
    fn_value = (p_array_T2[-1] / p_array_T2[T1_col_number]) * np.exp(first_term - second_term)
    return fn_value
a_value_T1 = AFunction(T_1)
print("A(T_1,T_2): ",a_value_T1)

def bondPriceT1():
    bond = np.zeros_like(T1_rate)
    for i in range(MC_paths):
        bond[i] = a_value_T1 * np.exp(-b_value_T1 * T1_rate[i])
    return bond
bond_price_array = bondPriceT1() #1D array of size MC_paths
print("bond price P(T1,T2) for each paths: ", bond_price_array)

#max payoff for each paths based on bond price at T1
payoffs_T1 = np.zeros_like(bond_price_array)
payoffs_T1 = np.maximum(bond_price_array-K,0) #np.maximum for element wise
#print(payoffs_T1)

#discount factor for each MC paths
discount_factor = np.zeros_like(bond_price_array)
for path in range(MC_paths):
    row = rates_MC_array[path,:]
    discount_factor[path] = np.exp(-np.sum(row[:T1_col_number+1]) * dt) #T1_col_number+1 to include that indexed value

option_price = np.sum(payoffs_T1 * discount_factor) / MC_paths
print("Discounted to present average option price: ",option_price)

def graphEverything():
    plt.plot(np.linspace(0,T_2,N), p_array_T2, label = 'P(0,T2) for all t')
    plt.plot(np.linspace(0,T_2,N), forward_array_T2, label = 'f(0,T2) for all t')
    plt.plot(np.linspace(0,T_2,N), theta_array, label = 'Theta(t)')
    for path in range(MC_paths):
        plt.plot(np.linspace(0,T_2,N), rates_MC_array[path,:])
    plt.legend()
    plt.xlabel("Each t in [0,T_2=1]")
    return plt.show()
#graphEverything()