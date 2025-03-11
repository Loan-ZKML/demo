mod onnx_converter;

use anyhow::{Result, Context};
use std::path::Path;
use std::process::Command;
use std::fs;

fn main() -> Result<()> {
    // Target address for proof generation
    let target_address = "0x276ef71c8F12508d187E7D8Fcc2FE6A38a5884B1";

    println!("Generating proof for address: {}", target_address);

    // Step 1: Generate feature data for target address
    // Using favorable features that will produce a good credit score
    let features = vec![0.8, 0.7, 0.6, 1.0];  // [tx_count, wallet_age, avg_balance, repayment_history]

    // Create directory for artifacts
    fs::create_dir_all("proof_generation")?;

    // Step 2: Create Python script for model generation
    create_model_script(&features, target_address)?;

    // Step 3: Run Python script to create ONNX model and input
    println!("Creating model and preparing input...");
    let status = Command::new("python3")
        .arg("create_model.py")
        .status()
        .context("Failed to execute Python script")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Model creation script failed with status: {}", status));
    }

    // Step 4: Run EZKL processing
    println!("Processing with EZKL...");
    let script_path = Path::new("run_ezkl.sh");

    // Write the EZKL script
    create_ezkl_script(script_path)?;

    // Make the script executable
    Command::new("chmod")
        .arg("+x")
        .arg(script_path)
        .status()
        .context("Failed to make script executable")?;

    // Run the script
    let status = Command::new("bash")
        .arg(script_path)
        .status()
        .context("Failed to execute EZKL script")?;

    if !status.success() {
        return Err(anyhow::anyhow!("EZKL script failed with status: {}", status));
    }

    println!("Proof generation complete!");
    println!("Generated artifacts:");
    println!(" - Model: proof_generation/credit_model.onnx");
    println!(" - Verification key: proof_generation/vk.key");
    println!(" - Proof: proof_generation/proof.json");
    println!(" - Verifier contract: proof_generation/Halo2Verifier.sol");
    println!(" - On-chain calldata: proof_generation/calldata.json");

    // Step 5: Copy artifacts to appropriate directories for the Solidity tests
    println!("Copying artifacts for Solidity tests...");

    // Ensure directories exist
    fs::create_dir_all("src")?;
    fs::create_dir_all("script")?;

    // Copy files
    fs::copy("proof_generation/Halo2Verifier.sol", "src/Halo2Verifier.sol")?;
    fs::copy("proof_generation/calldata.json", "script/calldata.json")?;
    fs::copy("proof_generation/proof.json", "script/proof.json")?;

    println!("Artifacts copied successfully!");

    Ok(())
}

fn create_model_script(features: &[f32], address: &str) -> Result<()> {
    let script = format!(r#"
import json
import numpy as np
import torch
import torch.nn as nn
from torch.onnx import export

# Test account address
test_address = "{}"

# Features for the test address
favorable_features = {:?}  # [tx_count, wallet_age, avg_balance, repayment_history]

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

print(f"Test address: {{test_address}}")
print(f"Features: {{favorable_features}}")
print(f"Calculated score: {{score:.4f}}")
print(f"Threshold for favorable rate: 0.5")
print(f"Qualifies for favorable rate (100% collateral): {{score > 0.5}}")

# Export to ONNX
export(
    model,
    input_tensor,
    "proof_generation/credit_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}}
)

# Create EZKL input
ezkl_input = {{
  "input_shapes": [[4]],
  "input_data": [favorable_features],
  "output_data": [[score]]
}}

with open("proof_generation/input.json", "w") as f:
    json.dump(ezkl_input, f, indent=2)

print("Model converted to ONNX and input prepared for EZKL")
"#, address, features);

    fs::write("create_model.py", script)?;
    Ok(())
}

fn create_ezkl_script(path: &Path) -> Result<()> {
    let script = r#"#!/usr/bin/env bash
set -e

cd proof_generation

# Check if EZKL is installed
if ! command -v ezkl &> /dev/null; then
    echo "EZKL not found. Please install it with: pip install ezkl"
    exit 1
fi

# Step 1: Generate settings
echo "Generating circuit settings..."
ezkl gen-settings -M credit_model.onnx -O settings.json

# Step 2: Calibrate settings
echo "Calibrating settings..."
ezkl calibrate-settings -M credit_model.onnx -D input.json -O settings.json

# Step 3: Compile model to circuit
echo "Compiling model to circuit..."
ezkl compile-circuit -M credit_model.onnx --compiled-circuit model.compiled -S settings.json

# Step 4: Download SRS if needed
if [ ! -f kzg.srs ]; then
    echo "Downloading SRS..."
    ezkl get-srs --srs-path kzg.srs
fi

# Step 5: Generate keys
echo "Running setup to generate keys..."
ezkl setup -M model.compiled --pk-path pk.key --vk-path vk.key --srs-path kzg.srs

# Step 6: Generate witness
echo "Generating witness..."
ezkl gen-witness -D input.json -M model.compiled -O witness.json

# Step 7: Generate proof
echo "Generating proof..."
ezkl prove --witness witness.json --proof-path proof.json --pk-path pk.key --compiled-circuit model.compiled --srs-path kzg.srs

# Step 8: Verify the proof locally
echo "Verifying proof locally..."
ezkl verify --proof-path proof.json --vk-path vk.key --srs-path kzg.srs

# Step 9: Generate Solidity verifier contract
echo "Generating Solidity verifier contract..."
ezkl create-evm-verifier --vk-path vk.key --sol-code-path Halo2Verifier.sol --srs-path kzg.srs

# Step 10: Generate calldata for on-chain verification
echo "Generating calldata for on-chain verification..."
ezkl encode-evm-calldata --proof-path proof.json --calldata-path calldata.json

echo "EZKL processing complete!"
"#;

    fs::write(path, script)?;
    Ok(())
}