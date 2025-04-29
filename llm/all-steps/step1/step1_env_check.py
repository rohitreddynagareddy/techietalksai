# import torch

# print("✅ PyTorch installation check")
# print(f"PyTorch version: {torch.__version__}")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🖥️ Using device: {device}")

# # Simple tensor example
# x = torch.tensor([1, 2, 3])
# print(f"🔢 Tensor x: {x}")
# print(f"📏 Shape of x: {x.shape}")

import torch
import platform
import sys

print("✅ PyTorch installation check")
print(f"PyTorch version: {torch.__version__}")

# Check if the system is running on Apple Silicon
machine_type = platform.machine()
is_apple_silicon = machine_type in ["arm64", "aarch64"]
print(f"🖥️ Machine type: {machine_type}")
print(f"🍎 Running on Apple Silicon: {is_apple_silicon}")

# Check if PyTorch is using the MPS backend
if is_apple_silicon:
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 PyTorch is using the MPS backend for Apple Silicon.")
    else:
        device = "cpu"
        print("⚠️ PyTorch is not using the MPS backend. Falling back to CPU.")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

# Simple tensor example
x = torch.tensor([1, 2, 3], device=device)
print(f"🔢 Tensor x: {x}")
print(f"Longitude of x: {x.shape}")