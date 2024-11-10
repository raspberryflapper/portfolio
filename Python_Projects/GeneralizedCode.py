import random
import numpy as np

def next_x0(x0_t_i):  # Calculates next iteration of x0
  K1 = h*((p_0 - q_0) * v_0 * x0_t_i - d_0 * x0_t_i)
  K2 = h*((p_0 - q_0) * v_0 * (x0_t_i + (K1/2)) - d_0 * (x0_t_i + (K1/2)))
  K3 = h*((p_0 - q_0) * v_0 * (x0_t_i + (K2/2)) - d_0 * (x0_t_i + (K2/2)))
  K4 = h*((p_0 - q_0) * v_0 * (x0_t_i + K3) - d_0 * (x0_t_i + K3))
  x0_t_next = x0_t_i + ((K1 + 2*K2 + 2*K3 + K4)/6)
  return x0_t_next

def x0_iter(Data):  # Iterates until gets to 14 days
  t_i = 0
  x0_t_i = Data * 0.01
  x0 = [x0_t_i]
  while t_i < 14:
    #print(f"Number of CSC cells after {t_i} days: {x0_t_i}.")
    x0_t_new = next_x0(x0_t_i)
    x0.append(x0_t_new)
    x0_t_i = x0_t_new
    t_i += h
  #print(f"After {t_i} days, there are {x0_t_i} CSC cells.")
  return x0

def next_x1(x0_t_i, x1_t_i):  # Calculates next iteration of x1
  K1 = h*((1 - p_0 + q_0) * v_0 * x0_t_i + (p_1 - q_1) * v_1 * x1_t_i - d_1 * x1_t_i)
  K2 = h*((1 - p_0 + q_0) * v_0 * x0_t_i + (p_1 - q_1) * v_1 * (x1_t_i + (K1/2)) - d_1 * (x1_t_i + (K1/2)))
  K3 = h*((1 - p_0 + q_0) * v_0 * x0_t_i + (p_1 - q_1) * v_1 * (x1_t_i + (K2/2)) - d_1 * (x1_t_i + (K2/2)))
  K4 = h*((1 - p_0 + q_0) * v_0 * x0_t_i + (p_1 - q_1) * v_1 * (x1_t_i + K3) - d_1 * (x1_t_i + K3))
  x1_t_next = x1_t_i + ((K1 + 2*K2 + 2*K3 + K4)/6)
  return x1_t_next

def x1_iter(Data):  # Iterates until gets to 14 days
  t_i = 0
  x0_t_i = Data * 0.01
  x1_t_i = Data * 0.095
  x1 = [x1_t_i]
  while t_i < 14:
    #print(f"Number of PC1 cells after {t_i} days: {x1_t_i}.")
    x1_t_new = next_x1(x0_t_i, x1_t_i)
    x1.append(x1_t_new)
    x1_t_i = x1_t_new
    t_i += h
  #print(f"After {t_i} days, there are {x1_t_i} PC1 cells.")
  return x1

def next_x2(x1_t_i, x2_t_i):  # Calculates next iteration of x2
  K1 = h*((1 - p_1 + q_1) * v_1 * x1_t_i + (p_2 - q_2) * v_2 * x2_t_i - d_2 * x2_t_i)
  K2 = h*((1 - p_1 + q_1) * v_1 * x1_t_i + (p_2 - q_2) * v_2 * (x2_t_i + (K1/2)) - d_2 * (x2_t_i + (K1/2)))
  K3 = h*((1 - p_1 + q_1) * v_1 * x1_t_i + (p_2 - q_2) * v_2 * (x2_t_i + (K2/2)) - d_2 * (x2_t_i + (K2/2)))
  K4 = h*((1 - p_1 + q_1) * v_1 * x1_t_i + (p_2 - q_2) * v_2 * (x2_t_i + K3) - d_2 * (x2_t_i + K3))
  x2_t_next = x2_t_i + ((K1 + 2*K2 + 2*K3 + K4)/6)
  return x2_t_next

def x2_iter(Data):  # Iterates until gets to 14 days
  t_i = 0
  x1_t_i = Data * 0.095
  x2_t_i = Data * 0.095
  x2 = [x2_t_i]
  while t_i < 14:
    #print(f"Number of PC2 cells after {t_i} days: {x2_t_i}.")
    x2_t_new = next_x2(x1_t_i, x2_t_i)
    x2.append(x2_t_new)
    x2_t_i = x2_t_new
    t_i += h
  #print(f"After {t_i} days, there are {x2_t_i} PC2 cells.")
  return x2

def next_x3(x2_t_i, x3_t_i):  # Calculates next iteration of x3
  K1 = h*((1 - p_2 + q_2) * v_2 * x2_t_i - d_3 * x3_t_i)
  K2 = h*((1 - p_2 + q_2) * v_2 * x2_t_i - d_3 * (x3_t_i + (K1/2)))
  K3 = h*((1 - p_2 + q_2) * v_2 * x2_t_i - d_3 * (x3_t_i + (K2/2)))
  K4 = h*((1 - p_2 + q_2) * v_2 * x2_t_i - d_3 * (x3_t_i + K3))
  x3_t_next = x3_t_i + ((K1 + 2*K2 + 2*K3 + K4)/6)
  return x3_t_next

def x3_iter(Data):  # Iterates until gets to 14 days
  t_i = 0
  x2_t_i = Data * 0.095
  x3_t_i = Data * 0.8
  x3 = [x3_t_i]
  while t_i < 14:
    #print(f"Number of TDC cells after {t_i} days: {x3_t_i}.")
    x3_t_new = next_x3(x2_t_i, x3_t_i)
    x3.append(x3_t_new)
    x3_t_i = x3_t_new
    t_i += h
  #print(f"After {t_i} days, there are {x3_t_i} TDC cells.")
  return x3

def x_total_t(Data):  # Adds all x's together for each t_i, returns a list for each step (t_i)
  x0 = x0_iter(Data)
  x1 = x1_iter(Data)
  x2 = x2_iter(Data)
  x3 = x3_iter(Data)
  x = []
  for i in range(0, len(x0)):
    x.append(x0[i] + x1[i] + x2[i] + x3[i])
  print(f"Number of total cells at time t_i via Runge Kutta Method: {x}")
  return x

def approx_function(data):  # The function we found that approximates the given data, returns a list for each step (t_i)
  a = data[0]
  b = data[1]
  c = data[2]
  t_i = 0
  est_func = []
  while t_i <= 14:
    f_t_i = a*np.exp(-b*np.exp(-c*t_i))
    est_func.append(f_t_i)
    t_i += h
  print(f"Number of total cells at time t_i via approximated function: {est_func}")
  return est_func

# Probabilities that cell of type i divides into two types i cells, two type (i+1) cells, synthesis rate for type i cells, and degradation rate for type i cells, respectively.
p_0 = random.uniform(0,1)
q_0 = random.uniform(0,1-p_0)
v_0 = random.uniform(0,100)
d_0 = random.uniform(0,0.001)
p_1 = random.uniform(0,1)
q_1 = random.uniform(0,1-p_1)
v_1 = random.uniform(0,100)
d_1 = random.uniform(0,0.001)
p_2 = random.uniform(0,1)
q_2 = random.uniform(0,1-p_2)
v_2 = random.uniform(0,100)
d_2 = random.uniform(0,0.001)
p_3 = random.uniform(0,1)
q_3 = random.uniform(0,1-p_3)
v_3 = random.uniform(0,100)
d_3 = random.uniform(0,0.6)
if p_0 + q_0 > 1 or p_1 + q_1 > 1 or p_2 + q_2 > 1 or p_3 + q_3 > 1:
  print("Error")

# Initial number of total cells (*normalized*) according to the given two sets of data
Data1 = 0.030577382985992482
Data2 = 0.03416739370426276

# Step-size
h = 0.5

x_total_t(Data = Data1)

# Parameters we found via Steepest Gradient Descent Method
a_1 = 1.2327906960529382
b_1 = 18.596939933504437
c_1 = 0.3406891505670266
data_1 = [a_1, b_1, c_1]

approx_function(data = data_1)

'''
a_2 = ?
b_2 = ?
c_2 = ?
data_2 = [a-2, b_2, c_2]
'''