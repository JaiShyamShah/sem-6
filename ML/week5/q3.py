import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First convolution + relu + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Second convolution + relu + pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output from the conv layers
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

epochs=5
# Training loop
for epoch in range(epochs):  # Run for 5 epochs
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

accuracy = (total_correct / total_samples) * 100
print(f'Test Accuracy: {accuracy:.2f}%')

print("\nCNN model parameters:")
# Assuming model is your CNNClassifier instance
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}")

# To calculate the total number of learnable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable parameters: {total_params}")
