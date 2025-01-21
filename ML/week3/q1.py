import torch
import matplotlib.pyplot as plt

# Training data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32)

# Initialize parameters
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Hyperparameters
learning_rate = 0.001
epochs = 1000

# Store loss values for plotting
loss_values = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = w * x + b
    loss = torch.mean((y - y_pred) ** 2)
    loss_values.append(loss.item())

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()

# Plot the graph of epoch vs loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch for Linear Regression')
plt.grid(True)
plt.show()

print(f"Final parameters: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"Final loss: {loss_values[-1]:.4f}")