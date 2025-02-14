use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use synthetic_data::model_trainer::CreditScoreModel;

pub struct EzklProcessor {
    model_path: String,
    input_path: String,
    output_dir: String,
}

impl EzklProcessor {
    pub fn new(model_path: &str, input_path: &str, output_dir: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            input_path: input_path.to_string(),
            output_dir: output_dir.to_string(),
        }
    }

    pub fn load_model(&self) -> Result<CreditScoreModel> {
        let model_json = fs::read_to_string(&self.model_path)
            .with_context(|| format!("Failed to read model file: {}", self.model_path))?;
        
        let model = serde_json::from_str(&model_json)
            .with_context(|| format!("Failed to parse model JSON: {}", self.model_path))?;
        
        Ok(model)
    }

    pub fn load_input(&self) -> Result<Value> {
        let input_json = fs::read_to_string(&self.input_path)
            .with_context(|| format!("Failed to read input file: {}", self.input_path))?;
        
        let input = serde_json::from_str(&input_json)
            .with_context(|| format!("Failed to parse input JSON: {}", self.input_path))?;
        
        Ok(input)
    }

    pub fn process(&self) -> Result<f64> {
        let model = self.load_model()?;
        let input = self.load_input()?;
        
        // Extract input features from JSON
        let input_features = self.extract_features_from_json(&input)
            .with_context(|| "Failed to extract features from input JSON")?;
        
        // Predict with the model
        let prediction = model.predict(&input_features)
            .with_context(|| "Failed to run prediction with model")?;
        
        // Generate proof configuration
        self.generate_proof_config(&input_features, prediction)?;
        
        Ok(prediction)
    }

    fn extract_features_from_json(&self, input: &Value) -> Result<Vec<f64>> {
        let features = input["features"].as_array()
            .with_context(|| "Input JSON does not contain 'features' array")?;
        
        let mut result = Vec::with_capacity(features.len());
        for feature in features {
            let value = feature.as_f64()
                .with_context(|| format!("Feature is not a number: {}", feature))?;
            result.push(value);
        }
        
        Ok(result)
    }

    fn generate_proof_config(&self, input_features: &[f64], prediction: f64) -> Result<()> {
        // Create a directory for output if it doesn't exist
        if !Path::new(&self.output_dir).exists() {
            fs::create_dir_all(&self.output_dir)
                .with_context(|| format!("Failed to create output directory: {}", self.output_dir))?;
        }
        
        // Create the proof configuration
        let config = json!({
            "model_path": self.model_path,
            "input": {
                "features": input_features,
            },
            "output": prediction,
            "settings": {
                "run_args": {
                    "method": "mock",
                    "num_proofs": 1
                }
            }
        });
        
        // Write the configuration to a file
        let config_path = format!("{}/proof_config.json", self.output_dir);
        fs::write(&config_path, serde_json::to_string_pretty(&config)?)
            .with_context(|| format!("Failed to write proof configuration to {}", config_path))?;
        
        Ok(())
    }
}

use synthetic_data::model_trainer::CreditScoreModel;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Processor for handling EZKL operations with the credit score model
pub struct EzklProcessor {
    model: CreditScoreModel,
}

#[derive(Serialize, Deserialize)]
pub struct EzklInput {
    pub input_data: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct EzklOutput {
    pub score: f64,
}

impl EzklProcessor {
    /// Create a new EzklProcessor with the given model
    pub fn new(model: CreditScoreModel) -> Self {
        Self { model }
    }

    /// Load a model from a JSON file
    pub fn load_model_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let model = serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        Ok(Self { model })
    }

    /// Process input data and return a credit score
    pub fn process(&self, input: &[f64]) -> f64 {
        self.model.predict(input)
    }

    /// Process input from a JSON file and save the output
    pub fn process_file<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input_path: P,
        output_path: Q,
    ) -> io::Result<()> {
        // Read input file
        let mut input_file = File::open(input_path)?;
        let mut contents = String::new();
        input_file.read_to_string(&mut contents)?;
        
        let input: EzklInput = serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        // Process the data
        let score = self.process(&input.input_data);
        
        // Save the output
        let output = EzklOutput { score };
        let output_json = serde_json::to_string_pretty(&output)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        let mut output_file = File::create(output_path)?;
        output_file.write_all(output_json.as_bytes())?;
        
        Ok(())
    }

    /// Generate a circuit configuration for EZKL
    pub fn generate_circuit_config<P: AsRef<Path>>(&self, output_path: P) -> io::Result<()> {
        // This is a placeholder for the actual EZKL circuit configuration generation
        // In a real implementation, this would create the necessary configuration for EZKL
        let config = serde_json::json!({
            "model_type": "linear_regression",
            "input_shape": [self.model.feature_count()],
            "output_shape": [1],
        });
        
        let config_str = serde_json::to_string_pretty(&config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        let mut config_file = File::create(output_path)?;
        config_file.write_all(config_str.as_bytes())?;
        
        Ok(())
    }
}

