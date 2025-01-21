import torch
x = torch.tensor(1.0, requires_grad=True)

f = torch.exp(-x**2 - 2*x - torch.sin(x))
f.backward()

print("Computed Gradient df/dx:", x.grad.item())

import math
analytical_gradient = torch.exp(-x**2 - 2*x - torch.sin(x)) * (-2*x - 2 - torch.cos(x))
print("Analytical Gradient df/dx:", analytical_gradient.item())