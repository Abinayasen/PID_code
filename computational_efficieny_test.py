import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import memory_profiler
from memory_profiler import profile


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Load MNIST data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define PID coefficients
P = 0.5
I = 0.2
D = 0.1

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# Initialize model and criterion
model_pid = SimpleNN()
criterion = nn.CrossEntropyLoss()

# Initialize previous update and momentum term
prev_update = [torch.zeros_like(param) for param in model_pid.parameters()]
momentum_term = [torch.zeros_like(param) for param in model_pid.parameters()]


# Define PID-inspired optimizer function with memory profiling
@profile
def train_pid():
    model_pid.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer_pid.zero_grad()
        output = model_pid(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()

        gradients = [param.grad for param in model_pid.parameters()]

        with torch.no_grad():
            for param, grad, prev_upd, mom_term in zip(model_pid.parameters(), gradients, prev_update, momentum_term):
                update = - (P * grad + I * grad.sum() + D * (grad - prev_upd))
                mom_term.copy_(momentum * mom_term + learning_rate * update)
                param.data.add_(mom_term)
                prev_upd.copy_(grad)

        optimizer_pid.step()


# Create PID-inspired optimizer
optimizer_pid = optim.SGD(model_pid.parameters(), lr=learning_rate, momentum=momentum)

# Run training with memory profiling for PID-inspired optimizer
train_pid()
