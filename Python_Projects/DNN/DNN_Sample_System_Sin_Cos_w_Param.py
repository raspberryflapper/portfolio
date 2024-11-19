'''
READ:
SeHwan Kim
11.19.2024

In this code, we use DNN (2 hidden layers) to solve a sample systems of ODE with 2 parameters.
The known system is:
\frac{dy_1}{dx} = \alpha * sin(x) <=> y1 = -\alpha * cos(x)
\frac{dy_2}{dx} = \beta* cos(x) <=> y2 = beta * sin(x)
'''

import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import scipy.integrate

# graph the appx. solns. and exact solns.
def GraphPlot(t,y1,y2):
    plt.plot(t,y1,label = "y1 (-alp * cos(x)) NN solution", color = "purple")
    plt.plot(t,y2,label = "y2 (beta * sin(x)) NN solution", color = "green")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return None

# set domain and discretize
t_start, t_end = 0.0, 6.0
m_points = 60
# reshape linspace to a 1D to 2D, with 1 column, with the reshape
disc_t = torch.linspace(t_start,t_end,m_points,requires_grad=True).reshape(-1,1)

# DNN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()

        # use 2 hidden layer
        # 1st layer with input dimension of 1 and 64 neurons
        self.fc1 = nn.Linear(1,64)
        # 2nd hidden layer with 64 neurons
        self.fc2 = nn.Linear(64,64)
        # output layer 64 to 2 for y_1 and y_2
        self.fc3 = nn.Linear(64,2)

        # Parameters to learn
        self.alpha = nn.Parameter(torch.tensor(-1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(5.0, requires_grad=True))

    # forward pass through the NN layers
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleDNN()
# Adam optimizer with step of 0.01
optimizer = opt.Adam(model.parameters(),lr=0.000005)

def trial_solution(t,nn_output):
    # nn_output is a tensor of shape (m_points, 2), where each row represents the outputs of the network for each t.
    # The first column ([:, 0]) is the predicted value for y_1,
    # and the second column ([:, 1]) is the predicted value for y_2.
    y1, y2 = nn_output[:, 0].reshape(-1,1), nn_output[:, 1].reshape(-1,1)
    y_1_pred = -5 + (t-t_start) * y1
    y_2_pred = (t-t_start) * y2
    return y_1_pred, y_2_pred

def loss_function(disc_t):
    nn_output = model(disc_t)
    y1_pred, y2_pred = trial_solution(disc_t,nn_output)
    dy1_dx_pred = torch.autograd.grad(y1_pred, disc_t, grad_outputs = torch.ones_like(y1_pred), create_graph=True)[0]
    dy2_dx_pred = torch.autograd.grad(y2_pred, disc_t, grad_outputs = torch.ones_like(y2_pred), create_graph=True)[0]

    alpha = model.alpha
    beta = model.beta

    f1 = alpha * torch.sin(disc_t)
    f2 = beta * torch.cos(disc_t)
    f1_data = -5.0 * torch.cos(disc_t)
    f2_data = 10 * torch.sin(disc_t)

    loss = nn.MSELoss()

    ODE_output_loss = loss(dy1_dx_pred,f1)
    ODE_output_loss_2 = loss(dy2_dx_pred,f2)
    ODE_total = ODE_output_loss + ODE_output_loss_2

    data_output_loss = loss(y1_pred,f1_data) #dy1_dx or y1_pred?
    data_output_loss_2 = loss(y2_pred,f2_data) #same as above
    data_total = data_output_loss + data_output_loss_2

    total = ODE_total + data_total
    return total

epochs = 100000
for epochs in range(epochs):
    optimizer.zero_grad()
    loss = loss_function(disc_t)
    loss.backward()
    optimizer.step()
    if epochs %10000==0:
        print(f"Epoch{epochs}, Loss:{loss.item()}")
        print(f"Alpha: {model.alpha}, Beta: {model.beta}")
    if abs(loss.item()) < 0.0001:
        print (f"Epoch{epochs}, loss: {loss.item()} has converged")
        break

with torch.no_grad():
    nn_output = model(disc_t)
    y_1_appx, y_2_appx = trial_solution(disc_t,nn_output)

x_vals = disc_t.detach().numpy()
y_vals_1 = y_1_appx.detach().numpy()
y_vals_2 = y_2_appx.detach().numpy()

# Plot NN solution vs Numerical solution
exact_soln_y1 = -5.0 * np.cos(disc_t.detach().numpy())
exact_soln_y2 = 10.0 * np.sin(disc_t.detach().numpy())
plt.plot(x_vals, exact_soln_y1, label='y1 (-alp*cos(x)) Exact', linestyle='--')
plt.plot(x_vals, exact_soln_y2, label='y2 (beta*sin(x)) Exact', linestyle=':')

GraphPlot(x_vals, y_vals_1, y_vals_2)