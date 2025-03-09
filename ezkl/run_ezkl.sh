#!/usr/bin/env bash
set -e

# Function to print error messages
error() {
    echo -e "\033[0;31mERROR:\033[0m $1" >&2
    exit 1
}

# Function to print success messages
success() {
    echo -e "\033[0;32mSUCCESS:\033[0m $1"
}

# Check if EZKL is installed
command -v ezkl >/dev/null 2>&1 || error "EZKL CLI is required but not installed. Please install from https://github.com/zkonduit/ezkl"

# Print EZKL version 
echo "Using EZKL version: $(ezkl --version 2>&1 || echo "Unknown")"

# Get base directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"
echo "Current working directory: $(pwd)"

# Print current directory contents for debugging
echo "Directory contents:"
ls -la

# Define paths for EZKL artifacts
JSON_MODEL_PATH="credit_model.json"
JSON_INPUT_PATH="input.json"
MODEL_PATH="credit_model.onnx"
INPUT_PATH="ezkl_input.json"
SETTINGS_PATH="settings.json"
COMPILED_PATH="model.compiled"
SRS_PATH="kzg.srs"
VK_PATH="vk.key"
PK_PATH="pk.key"
WITNESS_PATH="witness.json"
PROOF_PATH="proof.json"
VERIFIER_PATH="Halo2Verifier.sol"
CALLDATA_PATH="calldata.json"

# Ensure the required input files exist
if [ ! -f "$JSON_MODEL_PATH" ]; then
    error "$JSON_MODEL_PATH not found. Run the synthetic_data generator first."
fi

if [ ! -f "$JSON_INPUT_PATH" ]; then
    error "$JSON_INPUT_PATH not found. Run the synthetic_data generator first."
fi

echo "Starting EZKL processing with real EZKL binary..."

# Step 0: Convert JSON model to ONNX format
echo "Converting JSON model to ONNX format..."
cat > convert_model.py << 'EOF'
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import export

# Load model from JSON
def load_model_from_json(model_file):
    with open(model_file, 'r') as f:
        model_data = json.load(f)
    return model_data

# Load input data from JSON
def load_input_from_json(input_file):
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    return input_data

# Create a PyTorch model from the JSON coefficients
class LinearModel(nn.Module):
    def __init__(self, coefficients, intercept):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(len(coefficients), 1)
        
        # Set weights directly from coefficients
        weight = torch.tensor(coefficients).reshape(1, -1)
        self.linear.weight.data = weight
        
        # Set bias from intercept
        self.linear.bias.data = torch.tensor([intercept])
        
        # Add sigmoid to clamp output between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Main conversion function
def convert_to_onnx(model_file, input_file, onnx_output, ezkl_input_output):
    # Load the model and input data
    model_data = load_model_from_json(model_file)
    input_data = load_input_from_json(input_file)
    
    coefficients = model_data['coefficients']
    intercept = model_data['intercept']
    
    # Create PyTorch model
    model = LinearModel(coefficients, intercept)
    model.eval()
    
    # Get sample features from input
    sample_features = input_data['sample']['features']
    sample_input = torch.tensor([sample_features], dtype=torch.float32)
    
    # Export to ONNX
    export(
        model, 
        sample_input, 
        onnx_output,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Create EZKL input format
    with torch.no_grad():
        sample_output = model(sample_input)
    
    ezkl_input = {
        "input_shapes": [[len(sample_features)]],
        "input_data": [sample_features],
        "output_data": [sample_output.flatten().tolist()]
    }
    
    with open(ezkl_input_output, 'w') as f:
        json.dump(ezkl_input, f, indent=2)
    
    print(f"Model exported to {onnx_output}")
    print(f"EZKL input saved to {ezkl_input_output}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python convert_model.py <model_json> <input_json> <onnx_output> <ezkl_input_output>")
        sys.exit(1)
    
    model_file = sys.argv[1]
    input_file = sys.argv[2]
    onnx_output = sys.argv[3]
    ezkl_input_output = sys.argv[4]
    
    convert_to_onnx(model_file, input_file, onnx_output, ezkl_input_output)
EOF

python3 convert_model.py "$JSON_MODEL_PATH" "$JSON_INPUT_PATH" "$MODEL_PATH" "$INPUT_PATH" || error "Failed to convert model to ONNX"

# Step 1: Generate settings
echo "Generating circuit settings..."
ezkl gen-settings -M "$MODEL_PATH" -O "$SETTINGS_PATH"

# Step 2: Calibrate settings for better performance
echo "Calibrating settings..."
ezkl calibrate-settings -M "$MODEL_PATH" -D "$INPUT_PATH" -O "$SETTINGS_PATH"

# Step 3: Compile the model into a circuit
echo "Compiling model to circuit..."
ezkl compile-circuit -M "$MODEL_PATH" --compiled-circuit "$COMPILED_PATH" -S "$SETTINGS_PATH"

# Step 4: Download the Structured Reference String (SRS) if not present
if [ ! -f "$SRS_PATH" ]; then
    echo "Downloading SRS..."
    ezkl get-srs --srs-path "$SRS_PATH"
fi

# Step 5: Generate proving and verification keys
echo "Running setup to generate keys..."
ezkl setup -M "$COMPILED_PATH" --pk-path "$PK_PATH" --vk-path "$VK_PATH" --srs-path "$SRS_PATH"

# Step 6: Generate witness from input data
echo "Generating witness..."
ezkl gen-witness -D "$INPUT_PATH" -M "$COMPILED_PATH" -O "$WITNESS_PATH"

# Step 7: Generate proof
echo "Generating proof..."
ezkl prove --witness "$WITNESS_PATH" --proof-path "$PROOF_PATH" --pk-path "$PK_PATH" --compiled-circuit "$COMPILED_PATH" --srs-path "$SRS_PATH"

# Step 8: Verify the proof locally
echo "Verifying proof locally..."
ezkl verify --proof-path "$PROOF_PATH" --vk-path "$VK_PATH" --srs-path "$SRS_PATH"

# Step 9: Generate Solidity verifier contract
echo "Generating Solidity verifier contract..."
ezkl create-evm-verifier --vk-path "$VK_PATH" --sol-code-path "$VERIFIER_PATH" --srs-path "$SRS_PATH"

# Optional: Generate calldata for on-chain verification
echo "Generating calldata for on-chain verification..."
ezkl encode-evm-calldata --proof-path "$PROOF_PATH" --calldata-path "$CALLDATA_PATH"

success "ZKML credit scoring system setup complete!"
echo "Generated artifacts:"
echo " - Model: $MODEL_PATH"
echo " - Settings: $SETTINGS_PATH"
echo " - Circuit: $COMPILED_PATH"
echo " - Verification key: $VK_PATH"
echo " - Proving key: $PK_PATH"
echo " - Witness: $WITNESS_PATH"
echo " - Proof: $PROOF_PATH"
echo " - EVM verifier contract: $VERIFIER_PATH"
echo " - On-chain calldata: $CALLDATA_PATH"