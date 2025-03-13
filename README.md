# zkML Demo

[![Rust CI](https://github.com/loan-zkml/demo/actions/workflows/ci.yml/badge.svg)](https://github.com/loan-zkml/demo/actions/workflows/ci.yml)

## Project Overview

This project demonstrates the application of zero-knowledge machine learning (zkML) in privacy-preserving verification systems. It combines Rust implementations with Ethereum smart contracts to create secure, verifiable data processing.

### Core Components

- **synthetic_data**: Generation of synthetic data for testing and demonstration
- **ezkl**: Zero-knowledge proof generation and verification for machine learning models

## Continuous Integration

The project utilizes GitHub Actions for continuous integration with the following checks:

- **Build**: Ensures all workspace crates compile successfully
- **Test**: Runs the test suite across all crates
- **Check**: Verifies code without producing binaries
- **Clippy**: Enforces Rust's linting rules with no warnings

All these checks must pass for pull requests to be merged, maintaining code quality and project stability.

## Development

A Makefile is provided for convenient development:

```bash
# Build all crates
make

# Run tests
make test

# Run linting
make clippy

# See all available commands
make help
```

# ZKML Credit Scoring System for DeFi Loans

This project implements a privacy-preserving credit scoring system using zero-knowledge machine learning (ZKML) for DeFi loans.

## Overview

The system allows users to prove their creditworthiness without revealing their private financial data by using zero-knowledge proofs.

- **synthetic_data**: Library that provides synthetic credit data generation and model training
- **ezkl**: Main entry point that uses synthetic_data library and integrates with EZKL CLI
- **loan**: CLI interface for the system (in development)

## How It Works

1. Generate synthetic credit data and train a credit scoring model
2. Convert the model to ONNX format compatible with EZKL
3. Use EZKL to create zero-knowledge circuits and proofs
4. Generate a Solidity verifier contract for on-chain verification
5. Allow users to submit proofs to DeFi lending platforms for better loan terms

## Getting Started

### Prerequisites

- Rust toolchain
- Python with PyTorch and ONNX
- EZKL CLI tool installed (version 20.2.0+)

### Installation

```bash
# Clone the repository
git clone https://github.com/loan-zkml/demo.git
cd demo

# Install Python dependencies
pip install torch numpy onnx

# Install EZKL following instructions at
# https://github.com/zkonduit/ezkl
```

### Usage

```bash
# Run the complete pipeline from data generation to ZK proof creation
cd ezkl
cargo run
```

The `ezkl` crate serves as the main entry point for the application, internally using the `synthetic_data` library to:
1. Generate synthetic credit data
2. Train the ML model
3. Save model and sample input files
4. Process everything with the EZKL CLI to create ZK proofs

## Generated Artifacts

This project generates various data files and artifacts during execution, which are not tracked in Git:

- Model files: JSON and ONNX formats of the credit scoring model
- ZK artifacts: Proofs, keys, witness, and compiled circuits 
- Solidity contracts: Generated verifier for on-chain verification

See the .gitignore file for the complete list of untracked generated files.

## Documentation

For more detailed information, see [docs/overview.md](docs/overview.md).