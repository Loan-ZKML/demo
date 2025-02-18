//! Synthetic data generation and model training for credit score prediction.
//! 
//! This library provides functions for generating synthetic credit data and
//! training machine learning models for credit scoring. It's primarily used
//! by the ezkl crate in this project to create data for ZKML demonstrations.
//! 
//! # Example
//! 
//! ```no_run
//! use synthetic_data::{generate_synthetic_data, save_data_as_json};
//! use synthetic_data::model_trainer::CreditScoreModel;
//! use std::path::Path;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Generate synthetic data
//! let data = generate_synthetic_data(1000)?;
//! 
//! // Train model
//! let model = CreditScoreModel::train(&data)?;
//! 
//! // Export model and sample input
//! model.export_to_json(Path::new("credit_model.json"))?;
//! model.save_sample_input(&data, "input.json")?;
//! # Ok(())
//! # }
//! ```

pub mod model_trainer;
pub mod synthetic_data;

// Re-export commonly used items for convenience
pub use model_trainer::CreditScoreModel;
pub use synthetic_data::CreditData;
pub use synthetic_data::generate_synthetic_data;
pub use synthetic_data::save_data_as_json;


