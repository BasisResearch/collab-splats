# tests/conftest.py
import torch

# Disable TorchDynamo globally for tests to avoid PyTorch/Transformers import issues
torch._dynamo.disable()

# Optional: print a message in CI so you know this is active
print("⚡ TorchDynamo disabled for tests")
