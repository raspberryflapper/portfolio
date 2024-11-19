'''
READ:
SeHwan Kim
11.19.2024

In this code, we use DNN (2 hidden layers) to solve the systems of ODE from the
2023 REU project (figure 1c in the paper).
The known system, Figure 1c, is:

\frac{dx_0(t)}{dt} = (p_0-q_0) * v_0 * x_0(t) - d_0 *x_0(t)
\frac{dx_1(t)}{dt} = (1-p_0+q_0) * v_0 *x_0(t) + (p_1-q_1) * v_1 * x_1(t) - d_1 * x_1(t),
\frac{dx_2(t)}{dt} = (1-p_1+q_1) * v_1 * x_1(t) + (p_2-q_2) * v_2 * x_2(t) - d_2 * x_2(t),
\frac{dx_3(t)}{dt} = (1-p_2+q_2) * v_2 * x_2(t) - d_3 * x_3(t).
'''

import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import scipy.integrate

# graph the appx. NN solns. and plot the data.
def GraphPlot(t,x0,x1,x2,x3):
    plt.plot(t,y1,label = "y1 (prey) NN solution", color = "purple")
    plt.plot(t,y2,label = "y2 (predator) NN solution", color = "green")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return None

# lab data
data_1 = [(0,150000),
        (1,53777),
        (2,65333),
        (3,134909),
        (4,222000),
        (5,248773),
        (6,376560),
        (7,555000),
        (8,975000),
        (9,1280000),
        (12,2302000),
        (13,2673000),
        (14,2870000)]

# store y values (number of cells) of the data into array n
n = np.array([data[1] for data in data_1])
data = torch.tensor(n, dtype=torch.float32).reshape(-1, 1)

# set domain and discretize
t_start, t_end = 0.0, 14.0
m_points = 100
# reshape linspace to a 1D to 2D, with 1 column, with the reshape
disc_t = torch.linspace(t_start,t_end,m_points,requires_grad=True).reshape(-1,1)

# DNN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()
        self.fc1 = nn.Linear(1,128)
        # 2nd hidden layer with 128 neurons
        self.fc2 = nn.Linear(128,128)
        # output layer 128 to 2 for x0,...,x3
        self.fc3 = nn.Linear(128,4)
        
        # Parameters to learn
        self.p0 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.q0 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.v0 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.d0 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.p1 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.q1 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.v1 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.d1 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.p2 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.q2 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.v2 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.d2 = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.d3 = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    # forward pass through the NN layers
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleDNN()
# Adam optimizer with step of 0.01
optimizer = opt.Adam(model.parameters(),lr=0.00001)

def trial_solution(t,nn_output):
    # nn_output is a tensor of shape (m_points, 2), where each row represents the outputs of the network for each t.
    # The first column ([:, 0]) is the predicted value for y_1,
    # and the second column ([:, 1]) is the predicted value for y_2.
    x0, x1, x2, x3 = nn_output[:, 0].reshape(-1,1), nn_output[:, 1].reshape(-1,1), nn_output[:, 2].reshape(-1,1), nn_output[:, 3].reshape(-1,1)

    # the left side
    x0_pred = 150000 + (t-t_start) * x0
    x1_pred = 150000 + (t-t_start) * x1
    x2_pred = 150000 + (t-t_start) * x2
    x3_pred = 150000 + (t-t_start) * x3
    return x0_pred, x1_pred, x2_pred, x3_pred

def loss_function(disc_t):
    nn_output = model(disc_t)
    x0_pred, x1_pred, x2_pred, x3_pred = trial_solution(disc_t,nn_output)
    dx0_dt_pred = torch.autograd.grad(x0_pred, disc_t, grad_outputs = torch.ones_like(x0_pred), create_graph=True)[0]
    dx1_dt_pred = torch.autograd.grad(x1_pred, disc_t, grad_outputs = torch.ones_like(x1_pred), create_graph=True)[0]
    dx2_dt_pred = torch.autograd.grad(x2_pred, disc_t, grad_outputs = torch.ones_like(x2_pred), create_graph=True)[0]
    dx3_dt_pred = torch.autograd.grad(x3_pred, disc_t, grad_outputs = torch.ones_like(x3_pred), create_graph=True)[0]

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
    f1 = (p0-q0) * v0 * x0_pred - (d0 *x0_pred)
    f2 = (1-p0+q0) * v0 * x0_pred + (p1-q1) * v1 * x1_pred - d1 * x1_pred
    f3 = (1-p1+q1) * v1 * x1_pred + (p2-q2) * v2 * x2_pred - d2 * x2_pred
    f4 = (1-p2+q2) * v2 * x2_pred - d3 * x3_pred

    loss = nn.MSELoss()
    ODE_output_loss = loss(dx0_dt_pred,f1)
    ODE_output_loss1 = loss(dx1_dt_pred,f2)
    ODE_output_loss2 = loss(dx2_dt_pred,f2)
    ODE_output_loss3 = loss(dx3_dt_pred,f2)
    ODE_total = ODE_output_loss + ODE_output_loss1 + ODE_output_loss2 + ODE_output_loss3
    data_output_loss = loss(x0_pred,data)
    data_output_loss1 = loss(x1_pred,data)
    data_output_loss2 = loss(x2_pred,data)
    data_output_loss3 = loss(x3_pred,data)
    data_total = data_output_loss + data_output_loss1 + data_output_loss2 + data_output_loss3
    total = ODE_total + data_total
    return total

epochs = 150000
for epochs in range(epochs):
    optimizer.zero_grad()
    loss = loss_function(disc_t)
    loss.backward()
    optimizer.step()
    if epochs %10000==0:
        print (f"Epoch{epochs}, Loss:{loss.item()}")
    if abs(loss.item()) < 1e-6:
        print (f"Epoch{epochs}, loss: {loss.item()} has converged")
        break

with torch.no_grad():
    nn_output = model(disc_t)
    x_0_appx, x_1_appx, x_2_appx, x_3_appx, = trial_solution(disc_t,nn_output)

t_vals = disc_t.detach().numpy()
x_vals = x_0_appx.detach().numpy()
x_vals1 = x_1_appx.detach().numpy()
x_vals2 = x_2_appx.detach().numpy()
x_vals3 = x_3_appx.detach().numpy()

GraphPlot(t_vals, x_vals, x_vals1, x_vals2, x_vals3)