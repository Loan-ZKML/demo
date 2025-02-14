mod model_trainer;
mod synthetic_data;

use anyhow::Result;
use std::path::Path;

use model_trainer::CreditScoreModel;
use synthetic_data::{generate_synthetic_data, save_data_as_json};

fn main() -> Result<()> {
    // Generate synthetic data
    println!("Generating synthetic credit data...");
    let data = generate_synthetic_data(1000)?;
    save_data_as_json(&data, "credit_data.json")?;
    println!("Saved synthetic data to credit_data.json");

    // Train model
    println!("Training credit score model...");
    let model = CreditScoreModel::train(&data)?;

    // Export model to JSON
    println!("Exporting model to JSON format...");
    model.export_to_json(Path::new("credit_model.json"))?;

    // Save sample input for EZKL
    println!("Saving sample input for EZKL...");
    model.save_sample_input(&data, "input.json")?;

    println!("Done! Model exported to credit_model.json and sample input saved to input.json");
    println!("You can now proceed with EZKL to create ZK circuits and proofs.");

    Ok(())
}
