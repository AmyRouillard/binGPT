# %%

import torch
import torch_directml

# %%
# Initialize the DirectML device
device = torch_directml.device(torch_directml.default_device())
print(f"Using DirectML device: {device}")

# %%
# Create tensors and move them to the DirectML device
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# Perform a matrix multiplication on the DirectML GPU
z = x @ y
print("Matrix multiplication completed on DirectML!")

# %%
x.device.type, y.device.type, z.device.type

# %%
