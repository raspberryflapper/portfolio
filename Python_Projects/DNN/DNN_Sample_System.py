'''
READ:
SeHwan Kim
11.19.2024

This DNN ODE solver solves a known systems of ODE, Lotka-Volterra prey-predator model.

The known system is:
\frac{dy_1}{dx} = \alpha  *y_1 -\beta * y_1 * _y_2
\frac{dy_2}{dx} = \delta * y_1 * y_2 - \gamma * y_2

In this code, we use DNN (2 hidden layers) to solve,
where y_1(t) is the prey population and y_2(t) is the predator population.
'''

import torch
import numpy
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import scipy.integrate

# graph the appx. solns. and exact solns.
def GraphPlot(t,y1,y2):
    plt.plot(t,y1,label = "y1 (prey) NN solution", color = "purple")
    plt.plot(t,y2,label = "y2 (predator) NN solution", color = "green")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return None

# Define the Lotka-Volterra equations for the exact solution to compare NN solutions
def lotka_volterra(t, y):
    alpha = 1.0
    beta = 0.1
    gamma = 1.5
    delta = 0.075
    y1, y2 = y
    dydt = [alpha * y1 - beta * y1 * y2, delta * y1 * y2 - gamma * y2]
    return dydt

# set domain and discretize
t_start, t_end = 0.0, 6.0
m_points = 60
# reshape linspace to a 1D to 2D, with 1 column, with the reshape
disc_t = torch.linspace(t_start,t_end,m_points,requires_grad=True).reshape(-1,1)

# Numerical solution for reference using solve_ivp
y0 = [10, 5]  # Initial population of prey and predators
t_span = [t_start, t_end]
t_eval = disc_t.detach().numpy().flatten()

# solve the exact Lotka-Volterra equations using scipy IVP int
sol = scipy.integrate.solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval)

# Ensure numerical solutions are interpolated to match the discretized time points
y1_data = torch.tensor(sol.y[0], dtype=torch.float32).reshape(-1, 1)
y2_data = torch.tensor(sol.y[1], dtype=torch.float32).reshape(-1, 1)


# DNN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()

        # use 2 hidden layer
        # 1st layer with input dimension of 1 and 20 neurons
        self.fc1 = nn.Linear(1,128)
        # 2nd hidden layer with 20 neurons
        self.fc2 = nn.Linear(128,128)
        # output layer 20 to 2 for y_1 and y_2
        self.fc3 = nn.Linear(128,2)

        # Parameters to learn
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.delta = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    # forward pass through the NN layers
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleDNN()
# Adam optimizer with step of 0.01
optimizer = opt.Adam(model.parameters(),lr=0.0000125)

def trial_solution(t,nn_output):
    # nn_output is a tensor of shape (m_points, 2), where each row represents the outputs of the network for each t.
    # The first column ([:, 0]) is the predicted value for y_1,
    # and the second column ([:, 1]) is the predicted value for y_2.
    y1, y2 = nn_output[:, 0].reshape(-1,1), nn_output[:, 1].reshape(-1,1)
    y_1_pred = 10 + (t-t_start) * y1
    y_2_pred = 5 + (t-t_start) * y2
    return y_1_pred, y_2_pred

def loss_function(disc_t):
    nn_output = model(disc_t)
    y1_pred, y2_pred = trial_solution(disc_t,nn_output)
    dy1_dx_pred = torch.autograd.grad(y1_pred, disc_t, grad_outputs = torch.ones_like(y1_pred), create_graph=True)[0]
    dy2_dx_pred = torch.autograd.grad(y2_pred, disc_t, grad_outputs = torch.ones_like(y2_pred), create_graph=True)[0]

    alpha = model.alpha
    beta = model.beta
    gamma = model.gamma
    delta = model.delta

    f1 = alpha * y1_pred - (beta * y1_pred * y2_pred)
    f2 = delta * y1_pred * y2_pred - (gamma * y2_pred)

    loss = nn.MSELoss()
    ODE_output_loss = loss(dy1_dx_pred,f1)
    ODE_output_loss_2 = loss(dy2_dx_pred,f2)
    ODE_total = ODE_output_loss + ODE_output_loss_2
    data_output_loss = loss(y1_pred,y1_data)
    data_output_loss_2 = loss(y2_pred,y2_data)
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
        print (f"Epoch{epochs}, Loss:{loss.item()}")
    if abs(loss.item()) < 1e-8:
        print (f"Epoch{epochs}, loss: {loss.item()} has converged")
        break

with torch.no_grad():
    nn_output = model(disc_t)
    y_1_appx, y_2_appx = trial_solution(disc_t,nn_output)

x_vals = disc_t.detach().numpy()
y_vals_1 = y_1_appx.detach().numpy()
y_vals_2 = y_2_appx.detach().numpy()

# Plot NN solution vs Numerical solution
plt.plot(t_eval, sol.y[0], label='y1 (prey) Exact', linestyle='--')
plt.plot(t_eval, sol.y[1], label='y2 (predator) Exact', linestyle=':')


GraphPlot(x_vals, y_vals_1, y_vals_2)