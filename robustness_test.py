'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Hyperparameters for PID control
P = 0.5
I = 0.2
D = 0.1
momentum = 0.9

# Define a function to update weights using PID-inspired control with momentum
def pid_momentum_optimizer(parameters, gradients, prev_update, momentum_term, learning_rate):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term.copy_(momentum * mom_term + learning_rate * update)
            param.data.add_(mom_term)
            prev_upd.copy_(grad)
    return prev_update, momentum_term

# Initialize the model, criterion
input_size = X.shape[1]
hidden_size = 10
output_size = len(iris.target_names)

criterion = nn.CrossEntropyLoss()

# Different learning rates to test
learning_rates = [0.001, 0.01, 0.1, 1.0]

# Function to train and evaluate the model with PID-inspired optimizer
def train_and_evaluate_pid(learning_rate):
    model = SimpleNN(input_size, hidden_size, output_size)

    # Initialize previous updates and momentum terms
    prev_update = [torch.zeros_like(param) for param in model.parameters()]
    momentum_term = [torch.zeros_like(param) for param in model.parameters()]

    # Training loop
    num_epochs = 100
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
def train_and_evaluate_adam(learning_rate):
    model = SimpleNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 100
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

# Test the model with different learning rates for PID optimizer
print("Testing PID-inspired optimizer with different learning rates:")
for lr in learning_rates:
    loss, accuracy = train_and_evaluate_pid(lr)
    print(f'Learning Rate: {lr}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Test the model with different learning rates for Adam optimizer
print("\nTesting Adam optimizer with different learning rates:")
for lr in learning_rates:
    loss, accuracy = train_and_evaluate_adam(lr)
    print(f'Learning Rate: {lr}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Function to add Gaussian noise to the data
def add_noise(data, noise_factor=0.1):
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return torch.tensor(noisy_data, dtype=torch.float32)

# Add noise to the training data
noise_factor = 0.1
X_train_noisy = add_noise(X_train.numpy(), noise_factor)
X_test_noisy = add_noise(X_test.numpy(), noise_factor)

# Hyperparameters for PID control
P = 0.5
I = 0.2
D = 0.1
momentum = 0.9

# Define a function to update weights using PID-inspired control with momentum
def pid_momentum_optimizer(parameters, gradients, prev_update, momentum_term, learning_rate):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term.copy_(momentum * mom_term + learning_rate * update)
            param.data.add_(mom_term)
            prev_upd.copy_(grad)
    return prev_update, momentum_term

# Initialize the model, criterion
input_size = X.shape[1]
hidden_size = 10
output_size = len(iris.target_names)

criterion = nn.CrossEntropyLoss()

# Function to train and evaluate the model with PID-inspired optimizer
def train_and_evaluate_pid(X_train, y_train, X_test, y_test, learning_rate=0.01):
    model = SimpleNN(input_size, hidden_size, output_size)

    # Initialize previous updates and momentum terms
    prev_update = [torch.zeros_like(param) for param in model.parameters()]
    momentum_term = [torch.zeros_like(param) for param in model.parameters()]

    # Training loop
    num_epochs = 100
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

    # Training loop
    num_epochs = 100
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

    # Training loop
    num_epochs = 100
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

# Test the model with noisy data for PID optimizer
print("Testing PID-inspired optimizer with noisy data:")
loss, accuracy = train_and_evaluate_pid(X_train_noisy, y_train, X_test_noisy, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Test the model with noisy data for Adam optimizer
print("\nTesting Adam optimizer with noisy data:")
loss, accuracy = train_and_evaluate_adam(X_train_noisy, y_train, X_test_noisy, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Test the model with noisy data for SGD optimizer
print("\nTesting SGD optimizer with noisy data:")
loss, accuracy = train_and_evaluate_sgd(X_train_noisy, y_train, X_test_noisy, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
