import torch

# Create tensors
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

# Addition
c = a + b
print("Addition:", c)

# Subtraction
d = a - b
print("Subtraction:", d)

# Element-wise multiplication
e = a * b
print("Element-wise multiplication:", e)

# Element-wise division
f = a / b
print("Element-wise division:", f)

# Dot product
g = torch.dot(a, b)
print("Dot product:", g)

# Matrix multiplication
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = torch.matmul(A, B)
print("Matrix multiplication:", C)
