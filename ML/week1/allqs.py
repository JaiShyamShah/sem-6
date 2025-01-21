import torch

#Q1
print("\nq1")
# 1. Create a tensor (original tensor)
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original Tensor:")
print(tensor)

# 2. Reshaping the tensor to shape (3, 2)
reshaped_tensor = tensor.reshape(3, 2)
print("\nReshaped Tensor (3, 2):")
print(reshaped_tensor)

# 3. Viewing the tensor (similar to reshape, returns a new view on the original data)
viewed_tensor = tensor.view(3, 2)
print("\nViewed Tensor (3, 2):")
print(viewed_tensor)

# 4. Stacking tensors along a new dimension (axis 0)
tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([3, 4])
tensor3 = torch.tensor([5, 6])

stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
print("\nStacked Tensor (along a new dimension):")
print(stacked_tensor)

# 5. Squeezing the tensor to remove dimensions of size 1
tensor_with_singleton_dim = torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]])
print("\nOriginal Tensor with singleton dimensions:")
print(tensor_with_singleton_dim.shape)

squeezed_tensor = tensor_with_singleton_dim.squeeze()
print("\nSqueezed Tensor (removing singleton dimensions):")
print(squeezed_tensor.shape)

# 6. Unsqueezing the tensor (adding a dimension of size 1 at a specific axis)
unsqueezed_tensor_axis0 = tensor.unsqueeze(0)
unsqueezed_tensor_axis1 = tensor.unsqueeze(1)


print("\nUnsqueezed Tensor (axis 0):")
print(unsqueezed_tensor_axis0.shape)

print("\nUnsqueezed Tensor (axis 1):")
print(unsqueezed_tensor_axis1.shape)

#Q2
print("\nq2")

# Permuting dimensions of a tensor
tensor = torch.randn(2, 3, 4)
permuted_tensor = tensor.permute(2, 0, 1)  # Change the dimension order (2 -> 0 -> 1)
print("Permuted Tensor:\n", permuted_tensor)

#Q3
print("\nq3")

# Tensor indexing
tensor = torch.randn(5, 5)
print("Original Tensor:\n", tensor)
indexed_value = tensor[2, 3]  # Accessing element at row 2, column 3
print("Indexed Value (2, 3):", indexed_value)

# Slicing
slice_tensor = tensor[1:4, 1:4]  # Slicing rows 1-3 and columns 1-3
print("Sliced Tensor:\n", slice_tensor)

#Q4
print("\nq4")

import numpy as np

# Convert numpy array to tensor
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print("Tensor from Numpy Array:\n", tensor_from_numpy)

# Convert tensor back to numpy array
tensor_to_numpy = tensor_from_numpy.numpy()
print("Numpy Array from Tensor:\n", tensor_to_numpy)


#Q5
print("\nq5")

random_tensor = torch.randn(7, 7)
print("Random Tensor with shape (7, 7):\n", random_tensor)

#Q6
print("\nq6")

# Matrix multiplication requires compatible dimensions. We need to transpose the second tensor
random_tensor_2 = torch.randn(1, 7)
result_matrix_multiplication = torch.matmul(random_tensor, random_tensor_2.T)
print("Matrix Multiplication Result:\n", result_matrix_multiplication)

#Q7
print("\nq7")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor1_gpu = torch.randn(2, 3).to(device)
tensor2_gpu = torch.randn(2, 3).to(device)
print("Tensor 1 on GPU:\n", tensor1_gpu)
print("Tensor 2 on GPU:\n", tensor2_gpu)

#Q8
print("\nq8")
# Adjusting shapes for matrix multiplication (tensor1: (2, 3), tensor2: (3, 2))
tensor2_gpu_transposed = tensor2_gpu.T
matrix_multiplication_result = torch.matmul(tensor1_gpu, tensor2_gpu_transposed)
print("Matrix Multiplication on GPU:\n", matrix_multiplication_result)

#Q9
print("\nq9")
max_value = matrix_multiplication_result.max()
min_value = matrix_multiplication_result.min()
print("Max Value:", max_value)
print("Min Value:", min_value)

#Q10
print("\nq10")
max_index = matrix_multiplication_result.argmax()
min_index = matrix_multiplication_result.argmin()
print("Max Index:", max_index)
print("Min Index:", min_index)


#Q11
print("\nq11")
# Set the seed for reproducibility
torch.manual_seed(7)

# Create the tensor
tensor_1_1_1_10 = torch.randn(1, 1, 1, 10)
print("Original Tensor (1, 1, 1, 10):\n", tensor_1_1_1_10)
print("Shape of original tensor:", tensor_1_1_1_10.shape)

# Remove dimensions with size 1
tensor_removed_ones = tensor_1_1_1_10.squeeze()
print("Tensor after removing dimensions with size 1:\n", tensor_removed_ones)
print("Shape of new tensor:", tensor_removed_ones.shape)