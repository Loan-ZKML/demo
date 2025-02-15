mod processor;

use anyhow::Result;
use std::path::Path;

use synthetic_data::{generate_synthetic_data, save_data_as_json};
use synthetic_data::model_trainer::CreditScoreModel;
use processor::EzklProcessor;

fn main() -> Result<()> {
    // Step 1: Generate synthetic data
    println!("Generating synthetic credit data...");
    let data = generate_synthetic_data(1000)?;
    save_data_as_json(&data, "credit_data.json")?;
    println!("Saved synthetic data to credit_data.json");

    // Step 2: Train model
    println!("Training credit score model...");
    let model = CreditScoreModel::train(&data)?;

    // Step 3: Export model to JSON
    println!("Exporting model to JSON format...");
    model.export_to_json(Path::new("credit_model.json"))?;

    // Step 4: Save sample input for EZKL
    println!("Saving sample input for EZKL...");
    model.save_sample_input(&data, "input.json")?;

    // Step 5: Process with EZKL
    println!("\nStarting EZKL processing...");

    // Define paths for EZKL artifacts
    let model_path = Path::new("credit_model.json");
    let input_path = Path::new("input.json");
    let settings_path = Path::new("settings.json");
    let compiled_path = Path::new("model.compiled");
    let srs_path = Path::new("kzg.srs");
    let vk_path = Path::new("vk.key");
    let pk_path = Path::new("pk.key");
    let witness_path = Path::new("witness.json");
    let proof_path = Path::new("proof.json");
    let sol_path = Path::new("CreditVerifier.sol");

    // Generate settings
    EzklProcessor::generate_settings(model_path, settings_path)?;

    // Calibrate settings
    EzklProcessor::calibrate_settings(model_path, input_path, settings_path)?;

    // Compile model to circuit
    EzklProcessor::compile_model(model_path, compiled_path, settings_path)?;

    // Get SRS
    EzklProcessor::get_srs(srs_path)?;

    // Run setup
    EzklProcessor::setup(compiled_path, vk_path, pk_path, srs_path)?;

    // Generate witness
    EzklProcessor::generate_witness(input_path, compiled_path, witness_path)?;

    // Generate proof
    EzklProcessor::generate_proof(witness_path, compiled_path, pk_path, proof_path, srs_path)?;

    // Verify proof
    let is_valid = EzklProcessor::verify_proof(proof_path, vk_path, srs_path)?;
    println!("Proof verification result: {}", if is_valid { "Valid" } else { "Invalid" });

    // Generate EVM verifier contract
    EzklProcessor::create_evm_verifier(vk_path, settings_path, sol_path)?;

    println!("\nZKML credit scoring system setup complete!");
    println!("Generated artifacts:");
    println!(" - Model: credit_model.json");
    println!(" - Circuit: model.compiled");
    println!(" - Verification key: vk.key");
    println!(" - Proving key: pk.key");
    println!(" - Proof: proof.json");
    println!(" - EVM verifier contract: CreditVerifier.sol");

    Ok(())
}