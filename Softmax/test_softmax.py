import torch
from cuda_softmax_ext import softmax_fused

# Move to GPU
x = torch.tensor([1.0,2.0,3.0,0.5,-1.0,2.5,0.0,1.5], device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

# Call your fused softmax
softmax_fused(x, y)

print("Output:", y)
print("Sum:", y.sum().item())
