
import json
import numpy as np
import torch
import torch.nn as nn
from torch.onnx import export

# Test account address
test_address = "0x276ef71c8F12508d187E7D8Fcc2FE6A38a5884B1"

# Features for the test address
favorable_features = [0.8, 0.7, 0.6, 1.0]  # [tx_count, wallet_age, avg_balance, repayment_history]

# Define a simple model - linear with sigmoid
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Simple linear model with sigmoid activation
        self.linear = nn.Linear(4, 1)

        # Set weights directly - this is a simplified version of the model
        # We use weights that match a simple credit scoring formula
        weight = torch.tensor([[0.3, 0.2, 0.2, 0.3]]).float()
        self.linear.weight.data = weight
        self.linear.bias.data = torch.tensor([0.0])

        # Sigmoid to normalize output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Create the model
model = SimpleModel()
model.eval()

# Calculate score
input_tensor = torch.tensor([favorable_features], dtype=torch.float32)
with torch.no_grad():
    score = model(input_tensor).item()

print(f"Test address: {test_address}")
print(f"Features: {favorable_features}")
print(f"Calculated score: {score:.4f}")
print(f"Threshold for favorable rate: 0.5")
print(f"Qualifies for favorable rate (100% collateral): {score > 0.5}")

# Export to ONNX
export(
    model,
    input_tensor,
    "proof_generation/credit_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# Create EZKL input
ezkl_input = {
  "input_shapes": [[4]],
  "input_data": [favorable_features],
  "output_data": [[score]]
}

with open("proof_generation/input.json", "w") as f:
    json.dump(ezkl_input, f, indent=2)

print("Model converted to ONNX and input prepared for EZKL")
