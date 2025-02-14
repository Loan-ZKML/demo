//! Synthetic data generation and model training for credit score prediction.

pub mod model_trainer;
pub mod synthetic_data;

// Re-export commonly used items for convenience
pub use model_trainer::CreditScoreModel;
pub use synthetic_data::CreditData;
pub use synthetic_data::generate_synthetic_data;
pub use synthetic_data::save_data_as_json;


