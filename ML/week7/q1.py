import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations for the dataset (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load dataset using ImageFolder
train_data = datasets.ImageFolder(root='./data/cats_and_dogs_filtered/train', transform=transform)
test_data = datasets.ImageFolder(root='./data/cats_and_dogs_filtered/validation', transform=transform)

# Split train and validation (80% train, 20% validation)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Dataloader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define Neural Network Architecture
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  # Adjusted for input size (128x128)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes: cat and dog

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# L2 Regularization using Optimizer's Weight Decay (Method 1)
def train_with_weight_decay(model, train_loader, val_loader, epochs=10, weight_decay=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)  # L2 regularization

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%')


# L2 Regularization using Loop to Calculate L2 Norm (Method 2)
def train_with_manual_l2(model, train_loader, val_loader, epochs=10, l2_lambda=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate L2 norm and add it as penalty
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += l2_lambda * l2_norm  # Adding L2 regularization

            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%')


# Initialize model and train with L2 regularization using optimizer's weight decay
model_weight_decay = CatDogModel().to(device)
print("Training with weight decay (L2 Regularization)...")
train_with_weight_decay(model_weight_decay, train_loader, val_loader, epochs=10, weight_decay=1e-5)

# Initialize model and train with manual L2 regularization
model_manual_l2 = CatDogModel().to(device)
print("Training with manual L2 regularization...")
train_with_manual_l2(model_manual_l2, train_loader, val_loader, epochs=10, l2_lambda=1e-5)
