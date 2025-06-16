'''
A simple code to generate short rate interest rates under the
Vasicek model. It only runs one path.

06.14.2025
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

a = 0.1#speed of mean reversion
b = 0.05#long term of mean level
T = 1.0
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

rates_array = np.zeros(N)
rates_array[0] = 0.05 #initial rate
print(rates_array)

for time in range(1,N):
    rates_array[time] = rates_array[time-1] + a*(b - rates_array[time-1])*dt + (sigma*dW[time])
    rates_array[time] = np.round(rates_array[time],3)
print(rates_array)

plt.plot(np.linspace(0,1,252),rates_array, label = 'Vasicek model rates')
plt.xlabel("time")
plt.ylabel("rates in decimal")
plt.show()