import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input layer: 4 features -> 64 units
        self.fc2 = nn.Linear(64, 32)  # Hidden layer: 64 units -> 32 units
        self.fc3 = nn.Linear(32, 3)  # Output layer: 32 units -> 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for hidden layers
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer with no activation (use CrossEntropyLoss)
        return x

# Instantiate the model
model = NeuralNet()

# PID-inspired parameters
P = 0.5
I = 0.2
D= 0.1

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# Initialize previous weight update and momentum
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
