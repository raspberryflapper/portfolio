'''
Overview:
    Given stock price and strike price, we use Geometric Brownian Motion
    to simulate stock prices. Then, we use Longstaff Schwartz Monte Carlo
    algorithm recursively to determine the best excerise strategy.
    In this implementation, we determine such strategy for an American put option.

User Guide:
    Set your global variables such as the number of trading days (N), Monte Carlo simulation paths (MC_paths),
    riskless rate (mu), drift (sigma), and initial stock price and strike price (int_stock_price and strike_price respectively).

    Fix the seed number for reproducibility (at the very top). Once the seed number is fixed, the user
    must toggle on/off the line in "def simulateStockPrice(...)" accordingly.

The code will output the following:
    1) GBM MC stock price matrix. Size is the number of paths by number of trading days(N).
    e.g. "This is the GBM MC stock prices: ..."
    2) A matrix of Option cash flow given GBM strike prices. Same size as the above matrix.
    e.g. "Final option cash flow: ..."
    3) A float figure of the value of the put option.
    e.g. "This is the final option value:..."
    4) Optional: Toggle on and off GBM Stock prices graph. "graphStockPrice(disc_t,sim_stock_prices)" at the end.

https://github.com/raspberryflapper
https://www.linkedin.com/in/sehwanmkim/

06.02.2025
'''

import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# fix seed for reproducibility:
'''
seed = 66
rng = np.random.default_rng(seed) #now, use "rng.standard_normal", instead of "np.random.randn"
'''

# parameters:
int_stock_price = 1.00
mu = 0.06 #riskless 6% return
sigma = 0.2 #drift
T = 1 #domain
N = 10 #number of trading days - number of steps
dt = T/N
MC_paths = 8
strike_price = 1.10

# use GBM to simulate stock prices and return a 2D array with simulated stock prices for each paths in rows for each trading days in columns
def simulateStockPrice(paths,steps,initial_price,delta,mu,sigma):
    stock_price = np.zeros((paths,steps+1)) #MC_paths by N+1 zero arrays to hold stock prices
    stock_price[:,0] = initial_price #initialize stock prices on column 0

    ###REPRODUCIBILITY TOGGLE HERE:
    # for fixed seed, uncomment below and comment out the line after:
    #dW = rng.standard_normal(size=(paths,steps)) * np.sqrt(delta)
    # for unfixed seed, uncomment below and comment out above
    dW = np.random.randn(paths,steps) * np.sqrt(delta)

    for i in range (N):
        stock_price[:,i+1] = stock_price[:,i] + (mu * stock_price[:,i]*delta) + (sigma * stock_price[:,i] * dW[:,i])
    return np.round(stock_price,3)
sim_stock_prices = simulateStockPrice(MC_paths,N,int_stock_price,dt,mu,sigma)
print("This is the GBM MC stock prices: ",sim_stock_prices)

# graph the stock prices
def graphStockPrice(t,sp):
    for m in range(MC_paths):
        plt.plot(t,sp[m,:])
    plt.xlabel("Trading times (T)")
    plt.ylabel("GBM estimate stock prices")
    plt.grid()
    plt.show()
    return None
disc_t = np.linspace(0,T,N+1) #(start, finish, num=) - give num= equally spaced elements from start to finish.

cash_flow_matrix = np.zeros_like(sim_stock_prices)
def calCashFlow(sim_price,strk_price,N_1):
    cash_flow = strk_price - sim_price[:,N_1]
    for i in range(len(cash_flow)):
        if cash_flow[i]<0:
            cash_flow[i] = 0
    return cash_flow
cash_flow_matrix[:, N] = calCashFlow(sim_stock_prices, strike_price, N) #initialize the last column

# For N-1 ITM paths
def IdentifyPaths(stock,step):
    prev_step_price = stock[:,step-1]
    mask = prev_step_price < strike_price #returns a boolean array, size (MC_paths,), 1D array. e.g. [False, True, True,...] - note, this is not the same as 2D array, which would've been 1 by 8
    row_indices = np.where(mask)[0] #which paths, i.e., indices of rows
    col_indice = step - 1
    matching_indices = [(row,col_indice) for row in row_indices] #tuple
    return matching_indices

def getPrices(stock,itm_indices):
    price_index = np.zeros(len(itm_indices)) #1D
    for i in range(len(itm_indices)):
        price_index[i] = stock[itm_indices[i]]#store 2D elements, ITM prices, to a 1D array
    return price_index

def ConstructITMVector(sim_stock_prices,N):
    indices = IdentifyPaths(sim_stock_prices,N) #tuple
    indices_array = np.array(indices)
    row_num = indices_array[:,0]
    itm_prices = getPrices(sim_stock_prices,indices) #1D array with ITM paths stock prices
    itm_vector = np.zeros(MC_paths) #initialize the vector
    itm_vector[row_num] = itm_prices #overwrite vector with itm stock prices in the corresponding indices
    return itm_vector

def calcITMPayoff(strike,sim):
    payoff = strike - sim
    for i in range(len(payoff)):
        if payoff[i] == 1.1:
            payoff[i] = 0
    return payoff

def DiscountCashFlow(gbm,step,cflow_matrix):
    indices = IdentifyPaths(gbm,step) #tuple of rows where ITM (e.g. [(1, 2),...] - row 1, column 2)
    discount_cflow = np.zeros(MC_paths)
    delta_t = dt
    riskless_rate = -mu
    for (i,_) in indices: #(i,_) is a tuple unpacking; "_" is placeholder<=> we don't care about the second element in the tuple
        for future_step in range(step,N+1): #current time step to the end
            future_cf = cflow_matrix[i,future_step]
            if future_cf > 0:
                time_diff = (future_step - (step-1))
                discount_cflow[i] = future_cf * np.exp(riskless_rate*time_diff)
                break

    return discount_cflow
    '''
    index = IdentifyPaths(gbm,step) #tuple of rows where ITM (e.g. [(1, 2),...] - row 1, column 2)
    #print("IdentifyPaths in DiscountCashFlow (indices of ITM N-1 paths)", index)
    indices_array = np.array(index) #convert tuple to array
    #print(indices_array)
    row_num = indices_array[:,0] #ITM paths indices in array
    discount_cflow = np.zeros(MC_paths)
    discount_cflow[row_num] = 0.94176 * cflow[row_num]
    '''

def calcRegression(itmX,cflowY):
    X = itmX.reshape(-1,1) #reshape for sklearn - takes 2D
    Y = cflowY
    poly = PolynomialFeatures(degree=2, include_bias=False) # Create polynomial features: x and x^2 (no bias because LinearRegression adds intercept)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Get fitted coefficients
    intercept = model.intercept_
    coefficients = model.coef_
    #print("Estimated model:")
    #print(f"E[Y|X] = {intercept:.3f} + {coefficients[0]:.3f}*x + {coefficients[1]:.3f}*x^2\n")
    continuation_reg = intercept + coefficients[0]*X + coefficients[1]*X**2
    return np.round(continuation_reg,3)

def comparePayoffs(exer,cont,cflow):
    for i in range(len(exer)):
        if exer[i]<cont[i]:
            exer[i] = 0
        elif exer[i]>cont[i]:
            cflow[i] = 0
        else:
            None
    return exer,cflow

# recursively calculate the LSMC for American Put Option:
for step in range(N,1,-1):
    # Construct stock prices for ITM paths (N-1), i.e., vector X:
    prices_itm = ConstructITMVector(sim_stock_prices,step)
    #print("result of ConstructITMVector - prices_itm:", prices_itm)

    # Calculate payoffs for those ITM paths, vector for payoff for exercising now:
    exer_now = calcITMPayoff(strike_price,prices_itm)
    #print("result of calcITMPayoff - exer_now:", exer_now)

    future_cash_flow = cash_flow_matrix[:, step]#calculates cashflow for all paths in N step
    #print("future_cash_flow:",future_cash_flow)

    # Discount future cash flow of the ITM paths if not exercise right now - i.e., construct vector Y:
    disc_cash_Y = DiscountCashFlow(sim_stock_prices,step,cash_flow_matrix) #vector Y
    #print("disc_cash_Y:", disc_cash_Y)

    # Now, use vector X and Y to regress to get E(Y|X):
    continuation_cash_flow = calcRegression(prices_itm,disc_cash_Y) #2by1
    continuation_cash_flow = continuation_cash_flow.reshape(-1) #reshape to 1D (MC_paths,) shape for consistency with other arrays in the loop
    #print("E[Y|X] - continuation cash flow:", continuation_cash_flow)
    
    # Compare the cash flow payoffs: exercise now vs expected value of holding:
    prev_N_cash_flow,updated_N_cash_flow = comparePayoffs(exer_now,continuation_cash_flow,future_cash_flow)
    #print("prev_N_cash_flow (i.e. opt_exercise):", prev_N_cash_flow)
    #print("updated_N_cash_flow (i.e. cflow):", updated_N_cash_flow)

    cash_flow_matrix[:,step-1] = prev_N_cash_flow
    # set the rest of the row elements of the path of exercise at t=step equal 0
    for i in range(len(prev_N_cash_flow)):
        if prev_N_cash_flow[i] != 0: #if we are exercising,
            cash_flow_matrix[i,step:] = 0 #set the cash flow of the rest of the row of the path to 0
        else:
            None
    cash_flow_matrix[:,step] = updated_N_cash_flow

print("Final option cash flow: ",cash_flow_matrix) #print final option cash flow matrix

# calculate the value of the American put option - discount any cash flow for each path back to t=0 and sum them up and average over the number of paths.
def calcFinalValue(final_matrix):
    disc_array = np.zeros(MC_paths)
    for rows in range(MC_paths):
            for cols in range(N+1):
                if final_matrix[rows,cols] != 0:
                    disc_array[rows] = final_matrix[rows,cols] * np.exp(-mu*cols)
                else:
                    None
    #print(disc_array)
    total_sum = sum(disc_array)
    final_value = total_sum / MC_paths
    return final_value
final_option_value = calcFinalValue(cash_flow_matrix)
print("This is the final option value: ",final_option_value)
# comment out below if no visualization
#graphStockPrice(disc_t,sim_stock_prices)