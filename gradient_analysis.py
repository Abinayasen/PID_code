import torch

def pid_optimizer(parameters, gradients, prev_update, P, I, D, learning_rate, momentum_term):
    with torch.no_grad():
        for param, grad, prev_upd, mom_term in zip(parameters, gradients, prev_update, momentum_term):
            update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
            mom_term = mom_term * momentum + update * learning_rate
            param.data.add_(mom_term)
            prev_upd.copy_(grad)
    return parameters, prev_update, momentum_term

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Example neural network
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

# Initialize model and optimizer parameters
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
P = 0.5
I = 0.2
D = 0.1
learning_rate = 0.01
momentum = 0.9

# Initialize optimizer-specific parameters
prev_update = [torch.zeros_like(param) for param in model.parameters()]
momentum_term = [torch.zeros_like(param) for param in model.parameters()]

# Lists to store gradient norms
gradient_norms = []

# Initialize criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Generate random data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))

# Training loop with gradient logging
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Track gradient norms
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    gradient_norms.append(grad_norm)

    # Update parameters using PID-inspired optimizer
    model_params, prev_update, momentum_term = pid_optimizer(model.parameters(),
                                                            [param.grad for param in model.parameters()],
                                                            prev_update, P, I, D, learning_rate, momentum_term)

# Plot gradient norms over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), gradient_norms, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms over Training Epochs (PID-inspired optimizer)')
plt.grid(True)
plt.show()
