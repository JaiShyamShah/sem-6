import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a=torch.tensor([2.0],requires_grad=True)
b=torch.tensor([3.0],requires_grad=True)
x=2*a + 3*b
y=5*a*a + 3*b*b*b
z=2*x + 3*y
z.backward()

print("Computed Gradient dz/da:", a.grad.item())

analytical_gradient = 4 + 30 * a.item()
print("Analytical Gradient dz/da:", analytical_gradient)