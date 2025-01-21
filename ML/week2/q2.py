import torch
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

a = torch.sigmoid(w * x + b)

a.backward()

print("Computed Gradient da/dw:", w.grad.item())

sigmoid_value = torch.sigmoid(w * x + b).item()  # Calculate sigmoid(wx + b)
analytical_gradient = sigmoid_value * (1 - sigmoid_value) * x.item()
print("Analytical Gradient da/dw:", analytical_gradient)