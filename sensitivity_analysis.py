import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# PID PARAMETERS
P = 0.5
I = 0.2
D = 0.1

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# Define a function to update weights using PID-inspired control with momentum
def pid_momentum_optimizer(parameters, gradients, prev_update, momentum_term):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term.copy_(momentum * mom_term + learning_rate * update)
            param.data.add_(mom_term)
            prev_upd.copy_(grad)
    return parameters, prev_update, momentum_term
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'momentum': [0.9, 0.95, 0.99],
    'P': [0.1, 0.5, 1.0],
    'I': [0.01, 0.05, 0.1],
    'D': [0.001, 0.01, 0.1]
}

param_combinations = list(ParameterGrid(param_grid))

results = []


def train_model(params):
    global P, I, D, learning_rate, momentum
    P = params['P']
    I = params['I']
    D = params['D']
    learning_rate = params['learning_rate']
    momentum = params['momentum']

    model = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    batch_size = 16

    previous_weight_update = [torch.zeros_like(param) for param in model.parameters()]
    momentum_term = [torch.zeros_like(param) for param in model.parameters()]

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            model_parameters = list(model.parameters())
            model_parameters, previous_weight_update, momentum_term = pid_momentum_optimizer(
                model_parameters, [param.grad for param in model_parameters], previous_weight_update, momentum_term)

            model.zero_grad()
            total_loss += loss.item()

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
    return accuracy


for params in param_combinations:
    accuracy = train_model(params)
    results.append((params, accuracy))

# Plot results
fig, ax = plt.subplots()

Ps= np.array([result[0]['P'] for result in results])
accuracies = np.array([result[1] for result in results])

for Pp in np.unique(Ps):
    mask = Ps == Pp
    ax.plot(Ps[mask], accuracies[mask], label=f'Pp={Pp}')

ax.set_xlabel('P(parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Sensitivity Analysis of P(parameter)')
ax.legend()
plt.show()

fig, ax = plt.subplots()

Is= np.array([result[0]['I'] for result in results])
accuracies = np.array([result[1] for result in results])

for Ip in np.unique(Is):
    mask = Is == Ip
    ax.plot(Is[mask], accuracies[mask], label=f'Ip={Ip}')

ax.set_xlabel('I(parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Sensitivity Analysis of I(parameter)')
ax.legend()
plt.show()

fig, ax = plt.subplots()

Ds= np.array([result[0]['D'] for result in results])
accuracies = np.array([result[1] for result in results])

for Dp in np.unique(Ds):
    mask = Ds == Dp
    ax.plot(Ds[mask], accuracies[mask], label=f'Dp={Dp}')

ax.set_xlabel('D(parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Sensitivity Analysis of D(parameter)')
ax.legend()
plt.show()

fig, ax = plt.subplots()

moms= np.array([result[0]['momentum'] for result in results])
accuracies = np.array([result[1] for result in results])

for moma in np.unique(moms):
    mask = moms == moma
    ax.plot(moms[mask], accuracies[mask], label=f'moma={moma}')

ax.set_xlabel('momentum')
ax.set_ylabel('Accuracy')
ax.set_title('Sensitivity Analysis of momentum')
ax.legend()
plt.show()
