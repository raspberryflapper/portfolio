'''
This code uses GBM stock prices, Black-Scholes pricing of a European call option
for the said stock (prices), and Delta to dynamically delta hedge and maintain
a self-financing portfolio.

Currently (06.13.2025), it only runs on one MC path and results the following:

This is the BS price at 0:  10.450583572185565
cash balance 0 -53.23194410594107
option payoff:  3.061089999999993
portfolio_value at maturity date:  7.550884210327382
rep_error_mat:  -4.489794210327389

in which we can see that the error is fairly large. Next step is to run multiple MC paths and take the mean,
hence minimizing this large error.

06.13.2025
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S = 100 #initial stock price
K = 100 #strike price
T = 1.0
r = 0.05 #riskless rate
sigma = 0.2 #volatility 
N = 252
dt = T/N

# fix seed for reproducibility:
seed = 66
rng = np.random.default_rng(seed) #now, use "rng.standard_normal", instead of "np.random.randn"
###REPRODUCIBILITY TOGGLE:
# for fixed seed, uncomment below and comment out the line after:
dW = rng.standard_normal(size=(N)) * np.sqrt(dt)
# for unfixed seed, uncomment below and comment out above
#dW = np.random.randn(paths,steps) * np.sqrt(delta)

stock_prices_GBM = np.zeros(N)
stock_prices_GBM[0] = S

def simulateStockPrice(rate,sigma):
    for i in range (1,N):
        stock_prices_GBM[i] = stock_prices_GBM[i-1] * np.exp((rate - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*dW[i])

    return np.round(stock_prices_GBM,5)

stock_prices_GBM = simulateStockPrice(r,sigma)
#print(np.size(stock_prices_GBM))
#print("This is the GBM MC stock prices: ",stock_prices_GBM)

def black_scholes_pricing(T,vola,rate,S,K):
    bs_array = np.zeros_like(S)
    for days in range(N):
        mat_time = T - (days*dt)
        d_1 = (np.log(S[days]/K)+(rate+(1/2)*vola**2)*(mat_time)) / (vola*(np.sqrt(mat_time)))
        d_2 = d_1 - (vola * np.sqrt(mat_time))
        bs_array[days] = S[days] * norm.cdf(d_1) - (K * np.exp(-rate*(mat_time)) * norm.cdf(d_2))
    return bs_array

bs_pricing = black_scholes_pricing(T,sigma,r,stock_prices_GBM,K)
#print("This is the BS price: $", bs_pricing)
print("This is the BS price at 0: ",bs_pricing[0])

#plt.plot(np.linspace(0,1,252),stock_prices_GBM,label = 'stock GBM')
#plt.plot(np.linspace(0,1,252),bs_pricing,label = 'BS pricing')
#plt.show()

def calcDelta(S):
    step = 0.25
    delta_array = np.zeros_like(S)
    s_add = black_scholes_pricing(T,sigma,r,S+step,K)
    s_minus = black_scholes_pricing(T,sigma,r,S-step,K)
    delta_array = (s_add-s_minus) / (2*step)
    return delta_array

delta_values = calcDelta(stock_prices_GBM)
#print("This is the delta: ",delta_values)

hedge_cost_shares = np.zeros_like(stock_prices_GBM)
hedge_cost_shares = delta_values * stock_prices_GBM

#hedge_cost_changes = np.diff(hedge_cost_shares) #numpy.diff(array)=>output[i] = array[i+1]-array[i]

cash_balance = np.zeros_like(hedge_cost_shares)
cash_balance[0] = bs_pricing[0] - (delta_values[0]*stock_prices_GBM[0])
print("cash balance 0",cash_balance[0])
for time in range(1,N):
    cash_balance[time] = cash_balance[time-1]*np.exp(r*dt) - ((delta_values[time]*stock_prices_GBM[time])-(delta_values[time-1]*stock_prices_GBM[time-1]))

portfolio_value = hedge_cost_shares + cash_balance

option_payoff = np.max(stock_prices_GBM[-1]-K,0)
print("option payoff: ",option_payoff)
print("portfolio_value at maturity date: ",portfolio_value[-1])
rep_error_mat = option_payoff - portfolio_value[-1]
print("rep_error_mat: ",rep_error_mat)