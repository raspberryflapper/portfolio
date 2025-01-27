'''
READ:
SeHwan Kim
01.27.2025

In this code, we use DNN (2 hidden layers) to solve the systems of ODE with
undetermined coefficients from the 2023 REU project.
The known system, Figure 1c in the paper/project, is:

\frac{dx_0(t)}{dt} = (p_0-q_0) * v_0 * x_0(t) - (d_0 *x_0(t))
\frac{dx_1(t)}{dt} = (1-p_0+q_0) * v_0 * x_0(t) + (p_1-q_1) * v_1 * x_1(t) - (d_1 * x_1(t)),
\frac{dx_2(t)}{dt} = (1-p_1+q_1) * v_1 * x_1(t) + (p_2-q_2) * v_2 * x_2(t) - (d_2 * x_2(t)),
\frac{dx_3(t)}{dt} = (1-p_2+q_2) * v_2 * x_2(t) - (d_3 * x_3(t)).
'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt

real_data = torch.tensor([[0,150000],[1,60500],[2,98000],[3,185500],[4,222000],[5,234500],
                          [6,425500],[7,652500],[8,1068500],[9,1517500],[12,2938000],[13,2987500],
                          [14,2960500]],dtype=torch.float32)
# normalize the real world data
min_vals = real_data.min(dim=0)[0]
max_vals = real_data.max(dim=0)[0]
norm_real_data = (real_data - min_vals) / (max_vals - min_vals)
#print(f"this is the norm_real_data{norm_real_data}")
#print(f"last 2 of the first column eliminated:{norm_real_data[:-2,0]}")
# set domain and discretize
t_start, t_end = norm_real_data[0,0],norm_real_data[-1,0]
m_points = 10
# reshape linspace to a 1D to 2D, with 1 column, with the reshape
disc_t = torch.linspace(t_start,t_end,m_points,requires_grad=True).reshape(-1,1)

def InterpolateData(x_values,y_values,x):
    total = torch.zeros_like(x)
    n = x_values.size(0)
    for i in range(n):
        xi = x_values[i]
        yi = y_values[i]
        term = yi * torch.ones_like(x)
        for j in range(n):
            if i != j:
                xj = x_values[j]
                term *= (x-xj) / (xi-xj)
        total += term
    return total

# Lagrange interpolate the real world data set
interpolated_list = []
for i in range(len(disc_t)):
    interpolate_value = InterpolateData(norm_real_data[:,0],norm_real_data[:,1],disc_t[i])
    interpolated_list.append(interpolate_value.item())
interpolated_list_tensor = torch.tensor(interpolated_list) # 1 by m size
#print(f"this is the interpolated_list_tensor:{interpolated_list_tensor}")

# DNN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()
        self.fc1 = nn.Linear(1,64)
        self.fc2 = nn.Linear(64,64)
        # output layer 64 to 2 for x0,...,x3
        self.fc3 = nn.Linear(64,4)
        
        # Parameters to learn with initial guesses
        self.p0 = nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.q0 = nn.Parameter(torch.tensor(0.4, requires_grad=True))
        self.v0 = nn.Parameter(torch.tensor(7.40, requires_grad=True))
        self.d0 = nn.Parameter(torch.tensor(0.00003, requires_grad=True))
        self.p1 = nn.Parameter(torch.tensor(0.045, requires_grad=True))
        self.q1 = nn.Parameter(torch.tensor(0.40, requires_grad=True))
        self.v1 = nn.Parameter(torch.tensor(15.0, requires_grad=True))
        self.d1 = nn.Parameter(torch.tensor(0.0001, requires_grad=True))
        self.p2 = nn.Parameter(torch.tensor(0.16, requires_grad=True))
        self.q2 = nn.Parameter(torch.tensor(0.2, requires_grad=True))
        self.v2 = nn.Parameter(torch.tensor(7.0, requires_grad=True))
        self.d2 = nn.Parameter(torch.tensor(0.0003, requires_grad=True))
        self.d3 = nn.Parameter(torch.tensor(2.8, requires_grad=True))

    # forward pass through the NN layers
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleDNN()
optimizer = opt.Adam(model.parameters(),lr=0.0000025)

def trial_solution(t,nn_output):
    # nn_output is a tensor of shape (m_points, 4), where each row represents the outputs of the network for each t.
    # The first column ([:, 0]) is the predicted value for x_0 at each t_i,
    # and the second column ([:, 1]) is the predicted value for x_1.
    x0, x1, x2, x3 = nn_output[:, 0].reshape(-1,1), nn_output[:, 1].reshape(-1,1), nn_output[:, 2].reshape(-1,1), nn_output[:, 3].reshape(-1,1)

    # the left side
    # 150000 normalized is 0.0342
    x0_pred = 0.00342 + (t-t_start) * x0 # 10 percent of total initial
    x1_pred = 0.00171 + (t-t_start) * x1 # 5 percent
    x2_pred = 0.00171 + (t-t_start) * x2 # 5 percent
    x3_pred = 0.02736 + (t-t_start) * x3 # 80 percent
    return x0_pred, x1_pred, x2_pred, x3_pred

def loss_function(disc_t):
    nn_output = model(disc_t)
    x0_pred, x1_pred, x2_pred, x3_pred = trial_solution(disc_t,nn_output)
    # concatenate all the solutions into a single tensor
    x_pred_concat = torch.cat([x0_pred,x1_pred,x2_pred,x3_pred],dim=1) # size: m x 4. each column is the solution, each row is at discretized points t_i
    row_sum_x_pred_total = torch.sum(x_pred_concat,dim=1) # 1 by m - take sum across row (i.e. at each t_i, add all pred solutions up)
    # so the first column is the pred solutions summed at t_0, the second column pred solutions summed at t_1, and so on
    dx_pred = torch.zeros_like(x_pred_concat) # initialize the tensor with zeros
    # take auto grad and store the values to array mx4. This is the tensor of d\hat{y}/dt, essentially Jacobian or x_pred_concat
    for i in range(4):  # For each x_j
        dx_pred[:, i] = torch.autograd.grad(x_pred_concat[:, i], disc_t, grad_outputs=torch.ones_like(x_pred_concat[:, i]), create_graph=True)[0].squeeze()

    # parameters
    p0 = model.p0
    q0 = model.q0
    v0 = model.v0
    d0 = model.d0
    p1 = model.p1
    q1 = model.q1
    v1 = model.v1
    d1 = model.d1
    p2 = model.p2
    q2 = model.q2
    v2 = model.v2
    d2 = model.d2
    d3 = model.d3
    
    # the right side
    f = torch.zeros_like(x_pred_concat) # initialize tensor the same dimension as x_pred (i.e.m by 4)
    f[:,0] = (p0-q0) * v0 * x_pred_concat[:,0] - (d0 *x_pred_concat[:,0])
    f[:,1] = (1-p0+q0) * v0 * x_pred_concat[:,0] + (p1-q1) * v1 * x_pred_concat[:,1] - (d1 * x_pred_concat[:,1])
    f[:,2] = (1-p1+q1) * v1 * x_pred_concat[:,1] + (p2-q2) * v2 * x_pred_concat[:,2] - (d2 * x_pred_concat[:,2])
    f[:,3] = (1-p2+q2) * v2 * x_pred_concat[:,2] - (d3 * x_pred_concat[:,3])

    loss = nn.MSELoss()

    # calculate the ODE residual error
    ODE_residual = loss(dx_pred,f) # a scalar
    # calculate the data error
    data_output_loss = loss(row_sum_x_pred_total,interpolated_list_tensor)
    # calculate the total error
    total_error = ODE_residual + data_output_loss
    
    return total_error, row_sum_x_pred_total

epochs = 1000000
for epochs in range(epochs):
    optimizer.zero_grad()
    loss, x_pred_total = loss_function(disc_t)
    total_pred = x_pred_total.detach().numpy() #1 by m, each entry (column) is the sum of x's at each t_i
    loss.backward()
    optimizer.step()
    if epochs %20000==0:
        print(f"Epoch{epochs}, Loss:{loss.item()}, Total cells at each t_i:{total_pred}")
        print(f"Parameter: p0: {model.p0}")
        print(f"Parameter: q0: {model.q0}")
        print(f"Parameter: v0: {model.v0}")
        print(f"Parameter: d0: {model.d0}")
        print(f"Parameter: p1: {model.p1}")
        print(f"Parameter: q1: {model.q1}")
        print(f"Parameter: v1: {model.v1}")
        print(f"Parameter: d1: {model.d1}")
        print(f"Parameter: p2: {model.p2}")
        print(f"Parameter: q2: {model.q2}")
        print(f"Parameter: v2: {model.v2}")
        print(f"Parameter: d2: {model.d2}")
        print(f"Parameter: d3: {model.d3}")
    if abs(loss.item()) < 1e-4:
        print(f"Epoch{epochs}, loss: {loss.item()} has converged")
        break

with torch.no_grad():
    nn_output = model(disc_t)
    x_0_appx, x_1_appx, x_2_appx, x_3_appx, = trial_solution(disc_t,nn_output)
    print(f"FIANL PARAMETERS:\n")
    print(f"Parameter: p0: {model.p0}")
    print(f"Parameter: q0: {model.q0}")
    print(f"Parameter: v0: {model.v0}")
    print(f"Parameter: d0: {model.d0}")
    print(f"Parameter: p1: {model.p1}")
    print(f"Parameter: q1: {model.q1}")
    print(f"Parameter: v1: {model.v1}")
    print(f"Parameter: d1: {model.d1}")
    print(f"Parameter: p2: {model.p2}")
    print(f"Parameter: q2: {model.q2}")
    print(f"Parameter: v2: {model.v2}")
    print(f"Parameter: d2: {model.d2}")
    print(f"Parameter: d3: {model.d3}")

t_vals = disc_t.detach().numpy()
x_vals = x_0_appx.detach().numpy()
x_vals1 = x_1_appx.detach().numpy()
x_vals2 = x_2_appx.detach().numpy()
x_vals3 = x_3_appx.detach().numpy()
total_cells = x_vals + x_vals1 + x_vals2 + x_vals3
print(total_cells)

# graph the appx. NN solns. and plot the data.
def GraphPlot():
    plt.scatter(norm_real_data[:,0].detach().numpy(),norm_real_data[:,1].detach().numpy(),label = "2023 REU expData1", color = "Orange")
    plt.scatter(t_vals,total_cells,label = "DNN Total cells at discretized t_i's", color = "Blue")
    plt.xlabel("days")
    plt.ylabel("cell count")
    plt.legend()
    plt.show()

    return None
GraphPlot()

#####################################################################################################################################################
# real world lab data
'''
{Time(days) Total(cell number)
0 150000
1 60500
2 98000
3 185500
4 222000
5 234500
6 425500
7 652500
8 1068500
9 1517500
12 2938000
13 2987500
14 2960500

Time(days) Total(cell number)
0 150000
1 53777 
2 65333
3 134909
4 222000
5 248773
6 376560
7 555000
8 975000
9 1280000
12 2302000
13 2673000
14 2870000
'''