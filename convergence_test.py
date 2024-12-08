#PID1
#CONVERGENCE TEST

#importing the libraries
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real


import neural_networks_TensorFlow

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size= 0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

'''
class NeuralNet(nn.module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
#PID PARAMETERS
P = 0.5
I = 0.2
D = 0.1

#hyperparameters
learning_rate = 0.01
momentum = 0.9
previous_weight_update = [torch.zeros_like(param) for param in model.parameters()]
momentum_term = [torch.zeros_like(param) for param in model.parameters()]

# Define a function to update weights using PID-inspired control with momentum
def pid_momentum_optimizer(parameters, gradients, prev_update, momentum_term):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term = momentum * mom_term + learning_rate * update
            param += mom_term
            prev_upd = grad  # Update previous gradient for next iteration

    return parameters, prev_update, momentum_term


# Training loop with PID-inspired optimizer with momentum
num_epochs = 100
batch_size = 16

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights using PID-inspired optimizer with momentum
        with torch.no_grad():
            model_parameters = list(model.parameters())
            model_parameters, previous_weight_update, momentum_term = pid_momentum_optimizer(
                model_parameters, [param.grad for param in model_parameters], previous_weight_update, momentum_term)

        # Zero gradients manually
        model.zero_grad()

        total_loss += loss.item()

    # Print average loss for the epoch
    if (epoch+1) % 10 ==0 :
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(X_train_tensor)}')

# Evaluation on test set
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on test set: {accuracy:.2f}')


'''

#PID2 TEST:
'''
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = X.shape[1]
hidden_size = 10
num_classes = len(set(y))
model = SimpleNN(input_size, hidden_size, num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# PID optimizer parameters (example initial values)
momentum = 0.9


# Training function to be optimized
def pid_optimization_step_with_feedback(parameters, gradients, integral_error, prev_error, momentum_buffer, Kp, Ki, Kd,
                                        learning_rate):
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

    # Clamp the learning rate to stay within specified bounds
    learning_rate = max(0.0001, min(learning_rate, 1))

    # Update parameters with momentum
    for param, grad in zip(parameters, gradients):
        if param not in momentum_buffer:
            momentum_buffer[param] = torch.zeros_like(param)
        momentum_buffer[param] = momentum_buffer[param] * momentum + learning_rate * grad
        param.data -= momentum_buffer[param]

    return integral_error, current_error, learning_rate

start_time = time.time()
# Training function
def train_model(params):
    global model, integral_error, prev_error, momentum_buffer

    Kp, Ki, Kd, learning_rate = params
    integral_error = 0
    prev_error = 0
    momentum_buffer = {}
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and gradients computation
        loss.backward()
        gradients = [p.grad for p in model.parameters()]

        # PID optimization step
        integral_error, current_error, learning_rate = pid_optimization_step_with_feedback(
            model.parameters(), gradients, integral_error, prev_error, momentum_buffer,
            Kp, Ki, Kd, learning_rate
        )

        prev_error = current_error

        # Zero the gradients
        model.zero_grad()

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

    return -accuracy  # Negative accuracy because gp_minimize minimizes the objective
end_time = time.time()

# Define the search space for Bayesian optimization
space = [
    Real(0.01, 1.0, name='Kp'),
    Real(0.01, 1.0, name='Ki'),
    Real(0.01, 1.0, name='Kd'),
    Real(0.0001, 0.1, name='learning_rate')
]

# Optimize the training function
res = gp_minimize(train_model, space, n_calls=50, random_state=42)

print('Best accuracy: {:.2f}%'.format(-res.fun * 100))
print('Best parameters:', res.x)

# Train the final model with the best parameters
best_Kp, best_Ki, best_Kd, best_learning_rate = res.x
integral_error = 0
prev_error = 0
momentum_buffer = {}
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and gradients computation
    loss.backward()
    gradients = [p.grad for p in model.parameters()]

    # PID optimization step
    integral_error, current_error, learning_rate = pid_optimization_step_with_feedback(
        model.parameters(), gradients, integral_error, prev_error, momentum_buffer,
        best_Kp, best_Ki, best_Kd, best_learning_rate
    )

    prev_error = current_error

    # Zero the gradients
    model.zero_grad()

elapsed_time = end_time - start_time
print(f'Total elapsed time: {elapsed_time:.2f} seconds')
'''
#ADAM and SGD

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X.shape[1]
hidden_size = 10
num_classes = len(set(y))
model = SimpleNN(input_size, hidden_size, num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

'''
# Use ADAM optimizer
optimizer_adam = torch.optim.Adam(model.parameters())

start_time = time.time()
# Training function
def train_model_adam(num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer_adam.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer_adam.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test set with ADAM: {accuracy:.2f}')


end_time = time.time()
train_model_adam()

#SGD
# Use SGD optimizer
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

start_time = time.time()
# Training function
def train_model_sgd(num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer_sgd.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer_sgd.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test set with SGD: {accuracy:.2f}')
end_time = time.time()
train_model_sgd()

#RMSPROP
optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
start_time = time.time()
# Training function
def train_model_rmsprop(num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer_rmsprop.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer_rmsprop.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test set with RMSprop: {accuracy:.2f}')
end_time = time.time()
train_model_rmsprop()
'''
#adagrad
optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
start_time = time.time()
# Training function
def train_model_adagrad(num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer_adagrad.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer_adagrad.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy on test set with Adagrad: {accuracy:.2f}')
end_time = time.time()
train_model_adagrad()
elapsed_time = end_time - start_time
print(f'Total elapsed time: {elapsed_time:.2f} seconds')

