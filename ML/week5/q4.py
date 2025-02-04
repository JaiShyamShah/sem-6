# Modify CNN of Qn. 3 to reduce the number of parameters in the network. Draw a plot of
# percentage drop in parameters vs accuracy.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Modified CNN Model with fewer parameters
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Reduced number of filters in the conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Conv layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Conv layer
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling layer
        # Reduced the number of neurons in the fully connected layer
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First convolution + relu + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Second convolution + relu + pooling
        x = x.view(-1, 32 * 7 * 7)  # Flatten the output from the conv layers
        x = torch.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.fc2(x)  # Output layer
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop with tracking
epochs = 5
param_count = []
accuracy_values = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    accuracy_values.append(accuracy)

    # Track the number of parameters after each epoch
    total_params = sum(p.numel() for p in model.parameters())
    param_count.append(total_params)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Evaluate the model on the test set
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():  # Disable gradient computation for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

test_accuracy = (total_correct / total_samples) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Plot percentage drop in parameters vs accuracy
initial_params = sum(p.numel() for p in CNNClassifier().parameters())
param_drop_percentage = [(initial_params - count) / initial_params * 100 for count in param_count]

plt.plot(param_drop_percentage, accuracy_values)
plt.xlabel("Percentage Drop in Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Percentage Drop in Parameters vs Accuracy")
plt.grid(True)

plt.show()
print(f"\nTotal learnable parameters (after reduction): {total_params}")
