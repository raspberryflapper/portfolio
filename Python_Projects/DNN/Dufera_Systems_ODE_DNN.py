'''
READ:
SeHwan Kim
11.14.2024

This DNN ODE solver solves a known nonlinear systems of ODE found in
T. Dufera's paper (2021). The exact solutions are known, and the code
utilizes DNN to solve the system and plots the approximate solutions
against the exact solutions. Each functions loss is calculated with MSE
and the average of the two losses is taken to be the cost function. The DNN
uses 64 neurons for 2 hidden layers and ReLu as the activation function for both layers.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# plot and compare NN solution to the exact
def GraphPlot():
    plt.scatter(x_vals,y1_vals, label = "NN solution y1",color = 'Green')
    plt.scatter(x_vals,y2_vals, label = "NN solution y2",color = 'Blue')
    plt.plot(x_vals,exact_solution_y1(x_vals),label = "Exact solution y1 - sin(x)", color = 'Purple')
    plt.plot(x_vals,exact_solution_y2(x_vals),label = "Exact solution y2 - 1+x^2", color = 'Red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return None

x_start, x_end = 0.0, 1.0
m_points = 11
disc_x = torch.linspace(x_start,x_end,m_points, requires_grad=True).reshape(-1,1)

exact_solution_y1 = lambda x:np.sin(x)
exact_solution_y2 = lambda x:1+x**2

# Define NN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()
        # 1st layer
        self.fc1 = nn.Linear(1,64)
        # 2nd layer
        self.fc2 = nn.Linear(64,64)
        # output layer
        self.fc3 = nn.Linear(64,2)

    # forward pass through the NN layers
    def forward(self,x):
        # take first layer and activate with tanh
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = SimpleDNN()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

# trial solution - \hat{y}
def trial_solution(x,nn_output):
    y1_output, y2_output = nn_output[:, 0].reshape(-1, 1), nn_output[:, 1].reshape(-1, 1)
    y1_pred = (x-x_start) * y1_output
    y2_pred = 1 + (x-x_start) * y2_output
    return y1_pred, y2_pred

def loss_function(disc_x):
    # calculate N_j(t^i,P_j) - NN's output
    nn_output = model(disc_x)
    # set trial solution - prediction of y_n
    y_1_pred, y_2_pred = trial_solution(disc_x,nn_output)
    # approximation of NN of dy_n/dx
    dy1_dx_pred = torch.autograd.grad(y_1_pred, disc_x, grad_outputs = torch.ones_like(y_1_pred), create_graph=True)[0]
    dy2_dx_pred = torch.autograd.grad(y_2_pred, disc_x, grad_outputs = torch.ones_like(y_2_pred), create_graph=True)[0]
    f1_soln = torch.cos(disc_x)+y_1_pred**2+y_2_pred-(1+disc_x**2+torch.sin(disc_x)**2)
    f2_soln = 2*disc_x-(1+disc_x**2)*torch.sin(disc_x)+y_1_pred*y_2_pred
    loss = nn.MSELoss()
    loss_output = loss(dy1_dx_pred,f1_soln)
    loss_output_2 = loss(dy2_dx_pred,f2_soln)
    return torch.mean(loss_output+loss_output_2)

# Training loop
epochs  = 10000
for epochs in range(epochs):
    # reset the gradients as to not accumulate and use old gradients each loop
    optimizer.zero_grad()
    # calculate the cost
    loss = loss_function(disc_x)
    # compute the gradient of the loss wrt each parameter in the model (backward prop.)
    loss.backward()
    # update parameters (weights and biases)
    optimizer.step()

    # print loss every 1000 epochs by mod 500
    if epochs %1000==0:
        print(f"Epoch{epochs}, Loss:{loss.item()}") #Tensor.item() returns tensor as a standard Py number
    if abs(loss.item()) < 1e-18:
        print(f"Epoch{epochs}, loss converged.")
        break

# evaluate the TRAINED model
# with torch.no_grad() is a context manager to tell Torch not to trach operations
# saves on memory and speeds up computation
# use it for inference or validation
with torch.no_grad():
    nn_output = model(disc_x)
    y1_approx, y2_approx = trial_solution(disc_x, nn_output)

# convert to NumPy for plotting - numpy() is a Torch method fyi
# turn off grad tracking on disc_x before converting to numpy
x_vals = disc_x.detach().numpy()
y1_vals = y1_approx.numpy()
y2_vals = y2_approx.numpy()
# graph now the results.
GraphPlot()