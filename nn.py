# Make a 2 Layers NN

import torch
import numpy as np
import matplotlib.pyplot as plt

# device = torch.device("mps") # only for check gpu on my m1

# Define hyperparameters
input_size = 5  # Number of input neurons
hidden_size = 8  # Number of hidden neurons
output_size = 11  # Number of output neurons
learning_rate = 1e-4  # Learning rate for the optimizer (now constant or use a decay schedule)

# Self-defined input data and output data (x, y)
sample_size = 2000
batch_size = 20
num_epochs = int(sample_size/batch_size)  # Number of epochs for training
x = torch.randn(sample_size, input_size)
y = torch.randn(sample_size, output_size)

# Splitting Dataset
train_size = int(sample_size*0.8) # 80% of the data for training
test_size = sample_size - train_size  # 20% of the data for testing
train_dataset, test_dataset = torch.utils.data.random_split(torch.utils.data.TensorDataset(x, y), [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the neural network architecture
# inherit the torch.nn.Module
class MyNeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNeuralNet, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = self.hidden(x) # input layer to hidden layer (hidden layer)
        relu = self.relu(hidden) # activation function (hidden to output) (hidden layer)
        output = self.output(relu) # hidden to output (output layer)
        return output
        
model = MyNeuralNet(input_size, hidden_size, output_size)

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
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        y_pred = model(inputs) # forward pass
        loss = mse_loss_with_l2_regularization(y_pred, targets, model, l2_lambda) # compare the predict one and ans

        loss.backward() # backward pass
        
        optimizer.step()

        # reset the gradient to avoid accumulated gradient of next batch and epoch
        optimizer.zero_grad() 

# Testing pahase
# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        outputs = model(inputs)
        if batch_idx == 0:
            test_outputs = outputs
            test_targets = targets
        else:
            test_outputs = torch.cat((test_outputs, outputs), dim=0)
            test_targets = torch.cat((test_targets, targets), dim=0)

# Plot the testing output and target output
# y=x line
xx = np.linspace(test_targets.min(), test_targets.max(), 100)
yy = np.linspace(test_outputs.min(), test_outputs.max(), 100)
plt.plot(xx, yy, 'r')

plt.scatter(test_targets.detach().numpy(), test_outputs.detach().numpy())
plt.xlabel('Target Output')
plt.ylabel('Testing Output')
plt.show()
        


