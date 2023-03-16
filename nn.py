# Make a 2 Layers NN

import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu") # only for check gpu on my m1

# Define hyperparameters
# N is batch size(or data nums); D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# dimension as well as shape (e.g. 3D tensor with dimension (3,4,5))
# dimension here is flatten (e.g. (3,4,5) 3x4x5=60)
# so dimension as well as number of neurons
N, D_in, H, D_out = 500, 200, 11, 5
learning_rate = 1e-4  # Learning rate for the optimizer (now constant or use a decay schedule)

# Self-defined input data and output data (x, y)
# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = x.view(N, int(D_in/D_out), D_out)
y = torch.sum(y[:,:,:], dim=1)

num_epochs = 300  # Number of epochs

# Splitting dataset size (should split whole dataset)
train_size = int(N*0.8) # 80% of the data for training
test_size = N - train_size  # 20% of the data for testing
x_train, x_test = torch.split(x, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H), # input to hidden
        torch.nn.ReLU(), # hidden activation function
        torch.nn.Linear(H, D_out), # hidden to output
    ).to(device)

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Define the loss function  
# MSE(Mean Square Error) with L2 regularization (to avoid big weights and overfitting)
def mse_loss_with_l2_regularization(pred, ans, model, l2_weight):
    mse_loss = torch.nn.functional.mse_loss(pred, ans)
    l2_reg = 0.0 # store the sum of weight square
    for param in model.parameters():
        l2_reg += torch.sum(torch.square(param))
    loss = mse_loss + l2_weight * l2_reg
    return loss

# Training
l2_lambda = 0.01 # L2 weight of the weight penalty strngth
for t in range(num_epochs):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x_train)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the loss.
    loss = mse_loss_with_l2_regularization(y_pred, y_train, model, l2_lambda) # compare the predict one and ans

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward() # backward pass
    
    optimizer.step() # update weights (W)

    # reset the gradient to avoid accumulated gradient of next batch and epoch
    optimizer.zero_grad() 

# Testing pahase
# Evaluate the model on the test set
with torch.no_grad(): # means that the following context doesn't need to calculate gradient
    model.eval()
    for t in range(num_epochs):
        y_pred = model(x_test)
        if t == 0:
            test_outputs = y_pred
            test_targets = y_test
        else:
            test_outputs = torch.cat((test_outputs, y_pred), dim=0)
            test_targets = torch.cat((test_targets, y_test), dim=0)

# Plot the testing output and target output
# y=x line
xx = np.linspace(test_targets.min(), test_targets.max(), 100)
yy = np.linspace(test_outputs.min(), test_outputs.max(), 100)
plt.plot(xx, yy, 'r')

plt.scatter(test_targets.detach().numpy(), test_outputs.detach().numpy())
plt.xlabel('Target Output')
plt.ylabel('Testing Output')
plt.show()
        
# Save the model as a .pt file
torch.save(model.state_dict(), 'model.pt')