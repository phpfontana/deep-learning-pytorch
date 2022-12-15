# importing libraries
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Loading data
# 2. Model building
# 3. Training and evaluation
# 4. Saving and loading model

# ================================================================== #
#                     1. Loading data                                #
# ================================================================== #

# Setting hyper parameters
epochs = 200
learning_rate = 0.01

# 01. Data
# Known parameters
weight = 0.7
bias = 0.3

# Range of values for X
start = 0
end = 1
step = 0.2

# Creating X inputs
X = torch.arange(start, end, step).unsqueeze(dim=1)

# Creating y outputs
y = weight * X + bias

# Defining train/validation split
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

X_val, y_val = X[train_split:], y[train_split:]

# ================================================================== #
#                     2. Model Building                              #
# ================================================================== #

# Linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializing parameters with random weights
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # Forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias  # Linear regression


# Instance of the model
model = LinearRegression()

# ================================================================== #
#                     3. Training and Evaluation                     #
# ================================================================== #

# criterion and optimizer
criterion = nn.L1Loss()  # MAE loss 
optimizer = torch.optim.SGD(params=model.parameters(),  # Optimize model's parameters
                            lr=learning_rate)  # lr = Learning Rate

# Tracking different values
epoch_count = []
train_loss_values = []
val_loss_values = []

for epoch in range(epochs):  # Loop through the data
    # Training step
    model.train()

    # Forward pass
    y_train_pred = model(X_train)

    # Loss
    train_loss = criterion(y_train_pred, y_train)

    # Backward prop and optimizer
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()  # Perform gradient descent

    # Evaluation step
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():

        # Forward pass
        y_val_pred = model(X_val)

        # Loss
        val_loss = criterion(y_val_pred, y_val)

    # Printing training step and results
    if (epoch + 1) % 5 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        print(f"Epoch: {epoch + 1}/{epochs} | Train loss: {train_loss} | Val loss: {val_loss}")
        print(model.state_dict())
        print("\n")

# Plot loss curve
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, np.array(torch.tensor(val_loss_values).numpy()), label="Validation loss")
plt.title("Training and validation loss values")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# ================================================================== #
#                     4. Saving and Loading Model                    #
# ================================================================== #

# Save the model state dict
torch.save(model.state_dict(), 'linear-regression.pt')

# Instantiating new instance with model class
loaded_model = LinearRegression()

# Loading saved state_dict
loaded_model.load_state_dict(torch.load('linear-regression.pt'))
