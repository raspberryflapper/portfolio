'''
Use Black-Scholes PDE and the solution attained from transformting
the PDE into heat equation to approximate European call option. Then,
use the pricing value to obtain the greeks. Plot it against stock prices
and see how each of the greeks behave.

06.11.2025
'''

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

S = 100 #stock price at t
K = 90 #strike price
T = 1
t = 0.2
sigma = 0.20
int_rate = 0.06

def black_scholes_pricing(T,t,vola,rate,S,K):
    mat_time = T - t
    d_1 = (np.log(S/K)+(rate+(1/2)*vola**2)*(mat_time)) / (vola*(np.sqrt(mat_time)))
    d_2 = d_1 - vola * np.sqrt(mat_time)
    
    return S * norm.cdf(d_1) - (K * np.exp(-rate*(mat_time)) * norm.cdf(d_2))
bs_pricing = black_scholes_pricing(T,t,sigma,int_rate,S,K)
print("This is the BS price: $", bs_pricing)

def calcDelta():
    step = 2.5 #step h for S
    s_add = black_scholes_pricing(T,t,sigma,int_rate,S+step,K)
    s_minus = black_scholes_pricing(T,t,sigma,int_rate,S-step,K)
    return (s_add-s_minus) / (2*step)
print("This is the Delta: ",calcDelta())

def calcGamma():
    step = 2.5
    s_add = black_scholes_pricing(T,t,sigma,int_rate,S+step,K)
    s_minus = black_scholes_pricing(T,t,sigma,int_rate,S-step,K)
    return (s_add-2*bs_pricing+s_minus) / (step**2)
#print("This is the Gamma: ",calcGamma())

def calcVega():
    step = 0.05
    add_eps = black_scholes_pricing(T,t,sigma+step,int_rate,S,K)
    minus_eps = black_scholes_pricing(T,t,sigma-step,int_rate,S,K)
    return (add_eps - minus_eps) / (2*step)
#print("This is the Vega: ",calcVega())

def calcTheta():
    t_step = 1/365
    less_matur = black_scholes_pricing(T,t+t_step,sigma,int_rate,S,K)
    return (less_matur - bs_pricing) / t_step
#print("This is the Theta: ", calcTheta())
    
def calcRho():
    step = 0.00125
    rho_add = black_scholes_pricing(T,t,sigma,int_rate+step,S,K)
    rho_minus = black_scholes_pricing(T,t,sigma,int_rate-step,S,K)
    return (rho_add - rho_minus) / (2*step)
#print("This is the Rho: ",calcRho())

domain_s_prices = np.linspace(60,120,60)
def DeltaForGraph(domain):
    step = 2.5 #step h for S
    s_add = np.zeros_like(domain)
    s_minus = np.zeros_like(domain)
    for x in range(len(domain)):
        s_add[x] = black_scholes_pricing(T,t,sigma,int_rate,domain[x]+step,K)
        s_minus[x] = black_scholes_pricing(T,t,sigma,int_rate,domain[x]-step,K)
    return (s_add-s_minus) / (2*step)

def GammaForGraph(domain):
    step = 2.5 #step h for S
    s_add = np.zeros_like(domain)
    s_minus = np.zeros_like(domain)
    for x in range(len(domain)):
        s_add[x] = black_scholes_pricing(T,t,sigma,int_rate,domain[x]+step,K)
        s_minus[x] = black_scholes_pricing(T,t,sigma,int_rate,domain[x]-step,K)
    return (s_add-(2*bs_pricing)+s_minus) / (step**2)

def VegaForGraph(domain):
    step = 0.0025
    add_eps = np.zeros_like(domain)
    minus_eps = np.zeros_like(domain)
    for x in range(len(domain)):
        add_eps[x] = black_scholes_pricing(T,t,sigma+step,int_rate,domain[x],K)
        minus_eps[x] = black_scholes_pricing(T,t,sigma-step,int_rate,domain[x],K)
    return (add_eps - minus_eps) / (2*step)

def ThetaForGraph(domain):
    t_step = 5
    less_matur = np.zeros_like(domain)
    for x in range(len(domain)):
        less_matur[x] = black_scholes_pricing(T,t+t_step,sigma,int_rate,domain[x],K)
    return (less_matur - bs_pricing) / t_step

def RhoForGraph(domain):
    step = 0.00025
    rho_add = np.zeros_like(domain)
    rho_minus = np.zeros_like(domain)
    for x in range(len(domain)):
        rho_add[x] = black_scholes_pricing(T,t,sigma,int_rate+step,domain[x],K)
        rho_minus[x] = black_scholes_pricing(T,t,sigma,int_rate-step,domain[x],K)
    return (rho_add - rho_minus) / (2*step)

def graphGreeks(x,delta_y,gamma_y,vega_y,theta_y,rho_y):
    plt.plot(x,delta_y,label = 'Delta')
    #plt.plot(x,gamma_y,label = 'Gamma')
    plt.plot(x,vega_y,label = 'Vega')
    #plt.plot(x,theta_y,label = 'Theta')
    #plt.plot(x,rho_y, label = 'Rho')
    plt.xlabel("Stock prices in dollars ($)")
    plt.ylabel("Scalar values for each greeks")
    plt.legend()
    plt.show()
    return None
graphGreeks(domain_s_prices,DeltaForGraph(domain_s_prices),GammaForGraph(domain_s_prices),VegaForGraph(domain_s_prices),ThetaForGraph(domain_s_prices),RhoForGraph(domain_s_prices))