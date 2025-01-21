import torch
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
b = torch.tensor(3.0)
sig = torch.sigmoid(w * x + b)
sig.backward()
print("Computed Gradient ", w.grad.item())