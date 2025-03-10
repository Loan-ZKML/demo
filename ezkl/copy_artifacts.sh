#!/bin/bash

# Copy EZKL artifacts to the contracts project
# - Copies Halo2Verifier.sol to ../../contracts/src/
# - Copies calldata.json to ../../contracts/script/
# - Overwrites existing files

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting to copy EZKL artifacts to contracts project..."

# Check if source files exist
# Check if source files exist and display warnings if they don't, but don't exit
declare -a required_files=("Halo2Verifier.sol" "calldata.json" "updated_calldata.json" "verifier_abi.json" "vk.key" "updated_proof.json")
missing_files=false

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Warning: $file not found in current directory"
        missing_files=true
    fi
done

if [ "$missing_files" = true ]; then
    read -p "Some files are missing. Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting operation"
        exit 1
    fi
fi

# Check if destination directories exist
CONTRACT_SRC_DIR="../../contracts/src"
CONTRACT_SCRIPT_DIR="../../contracts/script"

if [ ! -d "$CONTRACT_SRC_DIR" ]; then
    echo "Error: Destination directory $CONTRACT_SRC_DIR does not exist"
    exit 1
fi

if [ ! -d "$CONTRACT_SCRIPT_DIR" ]; then
    echo "Error: Destination directory $CONTRACT_SCRIPT_DIR does not exist"
    exit 1
fi

# Copy Halo2Verifier.sol

echo "Copying Halo2Verifier.sol to $CONTRACT_SRC_DIR/"
cp "Halo2Verifier.sol" "$CONTRACT_SRC_DIR/" || {
    echo "Error: Failed to copy Halo2Verifier.sol"
    exit 1
}

# Copy calldata.json

echo "Copying calldata.json to $CONTRACT_SCRIPT_DIR/"
cp "calldata.json" "$CONTRACT_SCRIPT_DIR/" || {
    echo "Error: Failed to copy calldata.json"
    exit 1
}