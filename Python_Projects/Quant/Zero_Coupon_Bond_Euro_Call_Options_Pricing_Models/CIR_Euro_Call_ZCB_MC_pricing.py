'''
Use the MC interest rates derived from the CIR model to
estimate pricing for ZCB maturing at T_2. Then, price a
European call option on the said ZCB that matures at T_1 (T_1 < T_2).

07.02.2025
'''

import numpy as np
import matplotlib.pyplot as plt

# PARAMTERS
a = 0.5 #revert speed
b = 0.03 #long term mean
sigma = 0.1 #volatility
r0 = 0.03 #initial rate
T_2 = 1.0 #maturity of the ZCB
T_1 = 0.5#maturity of the (euro) call option
K = 0.9#strike price
N = 252 #trading days
MC_paths = 10 
dt = T_2 / N
seed = 70
rng = np.random.default_rng(seed)
dW = rng.standard_normal(size=(MC_paths,N)) * np.sqrt(dt) #for fixed seed
#dW =  np.random.randn(MC_paths,N) * np.sqrt(dt) #if not fixed

rates_MC_array = np.zeros((MC_paths,N))
rates_MC_array[:,0] = r0

def runMCRates(mc_array):
    for paths in range(MC_paths):
        for columns in range(1,N):
            rates_MC_array[paths,columns] = rates_MC_array[paths,columns-1] + a * (b - np.max(rates_MC_array[paths,columns-1]))*dt + (sigma * np.sqrt(np.max(rates_MC_array[paths,columns-1]))*dW[paths,columns])
    return mc_array
rates_MC_array = runMCRates(rates_MC_array)
#print(rates_MC_array)

def findColumnRate():
    rate = np.zeros(MC_paths)
    #column_numbers = np.zeros(MC_paths)
    for row in range(MC_paths):
        for n in range(N):
            step = n * dt
            if np.abs(step - T_1) < 0.0001:
                rate[row] = rates_MC_array[row,n]
                break
    return rate,n
T1_rate,T1_col_number = findColumnRate()
#print(T1_rate)

def discountFactors(rates):
    discount = np.zeros(MC_paths)
    for rows in range(MC_paths):
        discount[rows] = np.exp(-1*np.sum(rates[rows,])*dt)
    return discount
discount_paths = discountFactors(rates_MC_array)
#print(discount_paths)

def calcAFn(gam):
    mat_time = T_2 - T_1
    numerator = 2*gam*np.exp((a+gam)*(mat_time/2))
    denom = (a+gam)*(np.exp(gam*mat_time)-1)+(2*gam)
    value = (numerator / denom) ** ((2*a*b) / (sigma**2))
    return value

def calcBFn(gam):
    mat_time = T_2 - T_1
    numerator = 2*(np.exp(gam*mat_time)-1)
    denom = (a+gam)*(np.exp(gam*mat_time)-1)+(2*gam)
    value = numerator / denom
    return value

def exactPrice():
    price = np.zeros(MC_paths)
    gamma = np.sqrt((a**2)+2*sigma**2)
    price = calcAFn(gamma) * np.exp(-1 * calcBFn(gamma)*T1_rate)
    return price
analytic_price = exactPrice()
#print(analytic_price)

def computePayoff():
    max_element_wise = np.maximum(analytic_price-K,0)
    #print(max_element_wise)
    discount_factor = np.zeros_like(max_element_wise)
    for path in range(MC_paths):
        row = rates_MC_array[path,:]
        discount_factor[path] = np.exp(-np.sum(row[:T1_col_number+1]) * dt) #T1_col_number+1 to include that indexed value
    return np.mean(max_element_wise*discount_factor)
option_price = computePayoff()
print(option_price)