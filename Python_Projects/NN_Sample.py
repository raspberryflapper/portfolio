'''
We solve a sample problem of dy/dx = e^x using 1 hidden layer NN.

manully set seed:
# Set seed for reproducibility
torch.manual_seed(42)      # For PyTorch
np.random.seed(42)         # For NumPy
# Optionally, if you're using CUDA (GPU), you can also set the seed for that:
# torch.cuda.manual_seed_all(42)
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# plot and compare NN solution to the exact
def GraphPlot():
    plt.plot(x_vals,y_vals, label = "NN solution")
    plt.plot(x_vals,exact_solution(x_vals),label = "Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return None

# set exact solution to compare with NN solution
exact_solution = lambda x:np.exp(x)

# set domain
x_start, x_end = 0.0, 1.0
# set intial condtion, y(0.0) = e^(0.0)
initial_a = 1.0
# discretization points
m_points = 100
# discretize using torch - these are the inputs for our NN
disc_x = torch.linspace(x_start,x_end,m_points, requires_grad=True).reshape(-1,1)

# Define NN structure
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN,self).__init__()

        # use 1 hidden layer
        # 1st layer with input dimension of 1 and 10 neurons
        self.fc1 = nn.Linear(1,10)
        # 2nd, the output, layer with 10 neurons and 1 output for our dependent variable, y, in ODE
        self.fc2 = nn.Linear(10,1)

    # forward pass through the NN layers
    def forward(self,x):
        # take first layer and activate with tanh
        x = torch.tanh(self.fc1(x))
        # use tanh over ReLu for differentiability and smoothness when it comes do ODEs
        # send to the next layer - we don't want activation layer for the output layer
        # because that would restrict the output values, our appx.,  to the activation function's scale
        x = self.fc2(x)

        return x

model = SimpleDNN()
# Adam optimizer with step of 0.01
optimizer = optim.Adam(model.parameters(),lr=0.01)

# trial solution - \hat{y}
def trial_solution(x,nn_output):
    return initial_a + x * nn_output

def loss_function(disc_x):
    # calculate N_j(t^i,P_j) - NN's output
    nn_output = model(disc_x)
    # set trial solution - prediction of y_n
    y_pred = trial_solution(disc_x,nn_output)
    # approximation of NN of dy_n/dx
    dy_dx_pred = torch.autograd.grad(y_pred, disc_x, grad_outputs = torch.ones_like(y_pred), create_graph=True)[0]
    return torch.mean((dy_dx_pred-y_pred))**2

def train_model(seed):
    # set random seed for both torch and numpy - they have independent random generators.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training loop
    epochs  = 5000
    for epochs in range(epochs):
        # reset the gradients as to not accumulate and use old gradients each loop
        optimizer.zero_grad()
        # calculate the cost
        loss = loss_function(disc_x)
        # compute the gradient of the loss wrt each parameter in the model (backward prop.)
        loss.backward()
        # update parameters (weights and biases)
        optimizer.step()

        # print loss every 500 epochs by mod 500
        if epochs %500==0:
            print(f"Epoch{epochs}, Loss:{loss.item()}") #Tensor.sitem() returns tensor as a standard Py number

    return model, loss.item()

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


# evaluate the TRAINED model
# with torch.no_grad() is a context manager to tell Torch not to trach operations
# saves on memory and speeds up computation
# use it for inference or validation
with torch.no_grad():
        nn_output = model(disc_x)
        y_approx = trial_solution(disc_x, nn_output)

# convert to NumPy for plotting - numpy() is a Torch method fyi
# turn off grad tracking on disc_x before converting to numpy
x_vals = disc_x.detach().numpy()
y_vals = y_approx.numpy()
# graph now the results.
GraphPlot()