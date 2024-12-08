import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Define a simple neural network

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Generate large-scale random data
num_samples = 100000
input_size = 100
hidden_size = 50
output_size = 10

X_train = torch.randn(num_samples, input_size)
y_train = torch.randint(0, output_size, (num_samples,))
X_test = torch.randn(num_samples // 10, input_size)
y_test = torch.randint(0, output_size, (num_samples // 10,))

# Hyperparameters for PID control
P = 0.5
I = 0.2
D = 0.1
momentum = 0.9
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
# Define a function to update weights using PID-inspired control with momentum
def pid_momentum_optimizer(parameters, gradients, prev_update, momentum_term, learning_rate):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term.copy_(momentum * mom_term + learning_rate * update)
            param.data.add_(mom_term)
            prev_upd.copy_(grad)
    return prev_update, momentum_term

# Function to train and evaluate the model with PID-inspired optimizer
def train_and_evaluate_pid(X_train, y_train, X_test, y_test, learning_rate=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)

    # Initialize previous updates and momentum terms
    prev_update = [torch.zeros_like(param) for param in model.parameters()]
    momentum_term = [torch.zeros_like(param) for param in model.parameters()]

    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Gather gradients
        gradients = [param.grad for param in model.parameters()]

        # Update parameters using PID optimizer
        prev_update, momentum_term = pid_momentum_optimizer(
            model.parameters(), gradients, prev_update, momentum_term, learning_rate
        )

    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    return loss.item(), accuracy

# Function to train and evaluate the model with Adam optimizer
def train_and_evaluate_adam(X_train, y_train, X_test, y_test, learning_rate=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update parameters using Adam optimizer
        optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    return loss.item(), accuracy

# Function to train and evaluate the model with SGD optimizer
def train_and_evaluate_sgd(X_train, y_train, X_test, y_test, learning_rate=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update parameters using SGD optimizer
        optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)

    return loss.item(), accuracy

model = SimpleNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

# Test the model with large-scale data for PID optimizer
print("Testing PID-inspired optimizer with large-scale data:")
loss, accuracy = train_and_evaluate_pid(X_train, y_train, X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Test the model with large-scale data for Adam optimizer
print("\nTesting Adam optimizer with large-scale data:")
loss, accuracy = train_and_evaluate_adam(X_train, y_train, X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Test the model with large-scale data for SGD optimizer
print("\nTesting SGD optimizer with large-scale data:")
loss, accuracy = train_and_evaluate_sgd(X_train, y_train, X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
