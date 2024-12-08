import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error

# Load data
data = pd.read_csv(r"D:\CS\AI\DEEP LEARNING\hapiness_pro\test_file.csv")
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)  # Ensure Y is reshaped to (n_samples, 1)

sc = StandardScaler()
X = sc.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)



class HappinessDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


dataset = HappinessDataset(X, Y)
train_loader = DataLoader(dataset=dataset, batch_size=15, shuffle=True)  # Added batch_size
test_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=False)  # Added batch_size

# Parameters
input_size = X.shape[1]
hidden_layer1 = 10
hidden_layer2 = 5
hidden_layer3 = 2
output_size = 1
batch_size = 15
test_batch_size = 5
epochs = 100

# Define neural network
class Model(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, hidden_layer3, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.fc4 = nn.Linear(hidden_layer3, output_size)

        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        return out

# Create the network
net = Model(input_size, hidden_layer1, hidden_layer2, hidden_layer3, output_size)
criterion = torch.nn.MSELoss()

# PID optimization function
def pid_optimization_step_with_feedback(parameters, gradients, integral_error, prev_error, momentum_buffer, Kp, Ki, Kd,
                                        learning_rate, lr_min, lr_max, cyclic_lr_base, cycle_length, iteration):
    current_error = sum([(p.grad ** 2).sum().item() for p in parameters])  # Sum of squared gradients as the error

    # Compute PID terms
    proportional = current_error
    integral_error += current_error
    derivative = current_error - prev_error

    # Update learning rate using PID control
    if current_error > prev_error:
        learning_rate += (Kp * proportional + Ki * integral_error + Kd * derivative)
    else:
        learning_rate -= (Kp * proportional + Ki * integral_error + Kd * derivative)

    # Apply cyclic learning rate
    cycle_position = (iteration % cycle_length) / cycle_length
    learning_rate = cyclic_lr_base * (1 + np.sin(2 * np.pi * cycle_position))

    # Clamp the learning rate to stay within specified bounds
    learning_rate = max(lr_min, min(learning_rate, lr_max))

    # Update parameters with momentum
    for param, grad in zip(parameters, gradients):
        if param not in momentum_buffer:
            momentum_buffer[param] = torch.zeros_like(param)
        momentum_buffer[param] = momentum_buffer[param] * momentum + learning_rate * grad
        param.data -= momentum_buffer[param]

    return integral_error, current_error, learning_rate


# Parameters for PID optimization with momentum and decay
initial_lr = 0.1
Kp = 0.1
Ki = 0.01
Kd = 0.001
lr_min = 1e-5
lr_max = 1e-1
cyclic_lr_base = initial_lr
cycle_length = 200
momentum = 0.9

# Initialize PID variables
integral_error = 0
prev_error = 0
learning_rate = initial_lr
momentum_buffer = {}

# Create dataset and dataloaders
dataset = HappinessDataset(X, Y)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False)

# Training loop with PID optimization
for epoch in range(epochs):
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Zero the gradients manually
        net.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Perform PID optimization step
        gradients = [param.grad for param in net.parameters()]
        integral_error, prev_error, learning_rate = pid_optimization_step_with_feedback(
            net.parameters(), gradients, integral_error, prev_error, momentum_buffer, Kp, Ki, Kd,
            learning_rate, lr_min, lr_max, cyclic_lr_base, cycle_length, epoch * len(train_loader) + batch_idx
        )

    # Logging training progress
    if (epoch + 1) % 100 == 0:
        print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Learning Rate: {learning_rate:.6f}')

# Evaluation on test set
net.eval()
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = net(inputs)
        predictions.extend(outputs.squeeze().tolist() if outputs.dim() > 0 else [outputs.item()])
        actuals.extend(targets.squeeze().tolist() if targets.dim() > 0 else [targets.item()])

# Convert lists to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate R-squared
r2 = r2_score(actuals, predictions)
print(f'R-squared: {r2:.4f}')

# Calculate MSE
mse = mean_squared_error(actuals, predictions)
print(f'Mean Squared Error: {mse:.4f}')

# Print predictions and actuals for the first few examples
print("Predictions vs Actuals:")
for i in range(min(5, len(predictions))):
    print(f'Prediction: {predictions[i]:.4f}, Actual: {actuals[i]:.4f}')
