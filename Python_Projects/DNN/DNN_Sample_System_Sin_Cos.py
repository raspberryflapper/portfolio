'''
READ:
SeHwan Kim
11.19.2024

We solve a sample problem of:
dy_1/dx = sin(x) <-> y1 = -cos(x)
dy_2/dx = cos(x) <-> y2 = sin(x)
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# plot and compare NN solution to the exact
def GraphPlot():
    plt.scatter(x_vals,y1_vals, label = "NN solution - y1",color = 'Red')
    plt.scatter(x_vals,y2_vals, label = "NN solution - y2",color = 'Orange')
    plt.plot(x_vals,exact_solution_y1(x_vals),label = "Exact solution - y1: -cos(x)", color = 'Green')
    plt.plot(x_vals,exact_solution_y2(x_vals),label = "Exact solution - y2: sin(x)", color = 'Blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return None

# set domain
x_start, x_end = 0.0, 10.0
initial_y1 = -1 #-cos(0)
initial_y2 = 0

# discretization points
m_points = 40
# discretize using torch - these are the inputs for our NN
disc_x = torch.linspace(x_start,x_end,m_points, requires_grad=True).reshape(-1,1)

# set exact solutions to compare with NN solution
exact_solution_y1 = lambda x:np.cos(x) * -1
exact_solution_y2 = lambda x:np.sin(x)

# Define NN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()
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
optimizer = optim.Adam(model.parameters(),lr=0.00001)

# trial solution - \hat{y}
def trial_solution(x,nn_output):
    y1, y2 = nn_output[:, 0].reshape(-1, 1), nn_output[:, 1].reshape(-1, 1)
    trial_y1 = initial_y1 + (x-x_start) * y1
    trial_y2 = initial_y2 + (x-x_start) * y2

    return trial_y1, trial_y2

def loss_function(disc_x):
    # calculate N_j(t^i,P_j) - NN's output
    nn_output = model(disc_x)
    # set trial solution - prediction of y_n
    y1_pred, y2_pred = trial_solution(disc_x,nn_output)
    # approximation of NN of dy_n/dx
    dy1_dx_pred = torch.autograd.grad(y1_pred, disc_x, grad_outputs = torch.ones_like(y1_pred), create_graph=True)[0]
    dy2_dx_pred = torch.autograd.grad(y2_pred, disc_x, grad_outputs = torch.ones_like(y2_pred), create_graph=True)[0]
    f1_soln = torch.sin(disc_x)
    f2_soln = torch.cos(disc_x)
    MSE_loss = nn.MSELoss()
    loss_output_y1 = MSE_loss(dy1_dx_pred,f1_soln)
    loss_output_y2 = MSE_loss(dy2_dx_pred,f2_soln)
    return loss_output_y1, loss_output_y2

'''def train_model(seed):
    # set random seed for both torch and numpy - they have independent random generators.
    torch.manual_seed(seed)
    np.random.seed(seed)
'''
# Training loop
epochs  = 50000
for epochs in range(epochs):
    # reset the gradients as to not accumulate and use old gradients each loop
    optimizer.zero_grad()
    # calculate the cost
    loss_y1, loss_y2 = loss_function(disc_x)
    avg_loss = torch.mean(loss_y1+loss_y2)
    # compute the gradient of the loss wrt each parameter in the model (backward prop.)
    avg_loss.backward()
    # update parameters (weights and biases)
    optimizer.step()

    # print loss every 1000 epochs by mod 500
    if epochs %1000==0:
        print(f"Epoch{epochs}, Loss:{avg_loss.item()}") #Tensor.item() returns tensor as a standard Py number
    if abs(avg_loss.item()) <0.000001:
        print(f"Epoch{epochs}, loss converged.")
        break
'''
#   return model, loss.item()

# initialize variables for searching for the best model and seed
best_loss = float('inf') # initialize to a very large number (symbolic)
best_model = None
best_seed = None

# start training
for seed in range(5):
    print(f"\nTraining with {seed} seed")
    model, final_loss = train_model(seed)

    # compare the final loss and best model
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model
        best_seed = seed

torch.save(best_model.state_dict(), 'best_model.pth')
with open('best_seed.txt', 'w') as f:
    f.write(str(best_seed))

print(f"\n Best model seed: {best_seed}, best loss: {best_loss}")
'''

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