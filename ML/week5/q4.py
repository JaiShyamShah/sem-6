# Modify CNN of Qn. 3 to reduce the number of parameters in the network. Draw a plot of
# percentage drop in parameters vs accuracy.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class CNNClassifier(nn.Module):
    def __init__(self, conv1_out_channels=32, conv2_out_channels=64, fc1_out_features=20):
        super(CNNClassifier, self).__init__()
        # Modified: Removed third convolutional layer
        self.net = nn.Sequential(
            nn.Conv2d(1, conv1_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(conv2_out_channels*5*5, fc1_out_features, bias=True),  # Adjusted the size
            nn.ReLU(),
            nn.Linear(fc1_out_features, 10, bias=True)
        )

    def forward(self, x):
        features = self.net(x)
        features = features.view(features.size(0), -1)  # Flatten the features for fully connected layers
        return self.classification_head(features)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def experiment_with_parameters():
    results = []
    filter_configs = [
        (32, 64, 20),  # Reduced parameters
        (64, 128, 40),  # More filters
        (128, 256, 80),  # Even more filters
    ]

    for conv1_out_channels, conv2_out_channels, fc1_out_features in filter_configs:
        model = CNNClassifier(conv1_out_channels, conv2_out_channels, fc1_out_features)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, train_loader, criterion, optimizer, epochs=5)

        accuracy = evaluate(model, test_loader)

        num_params = count_parameters(model)

        results.append({
            'accuracy': accuracy,
            'num_params': num_params,
            'conv1_out_channels': conv1_out_channels,
            'conv2_out_channels': conv2_out_channels,
            'fc1_out_features': fc1_out_features
        })

    return results


results = experiment_with_parameters()

accuracies = [result['accuracy'] for result in results]
param_counts = [result['num_params'] for result in results]

initial_params = param_counts[0]
param_drops = [100 * (initial_params - p) / initial_params for p in param_counts]

plt.figure(figsize=(10, 6))
plt.plot(param_drops, accuracies, marker='o', linestyle='-', color='b')
plt.title('Percentage Drop in Parameters vs Accuracy (After Removing a Layer)')
plt.xlabel('Percentage Drop in Parameters')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()
