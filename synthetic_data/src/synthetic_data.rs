use anyhow::Result;
use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct CreditData {
    pub features: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
    pub feature_names: Vec<String>,
    // Optional mapping from Ethereum addresses to indices in the features/scores arrays
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address_mapping: Option<HashMap<String, usize>>,
}

pub fn generate_synthetic_data(num_samples: usize) -> Result<CreditData> {
    let mut rng = thread_rng();

    // Define distributions for different features
    let tx_count_dist = Uniform::new(0.0, 5000.0);
    let wallet_age_dist = Uniform::new(0.0, 365.0 * 5.0);
    let avg_balance_dist = Uniform::new(0.0, 10000.0);
    let repayment_hist_dist = Uniform::new(0.0, 1.0);

    // Generate random features
    let tx_count =
        Array1::from_iter((0..num_samples).map(|_| tx_count_dist.sample(&mut rng) / 5000.0));
    let wallet_age = Array1::from_iter(
        (0..num_samples).map(|_| wallet_age_dist.sample(&mut rng) / (365.0 * 5.0)),
    );
    let avg_balance =
        Array1::from_iter((0..num_samples).map(|_| avg_balance_dist.sample(&mut rng) / 10000.0));
    let repayment_hist = Array1::from_iter(
        (0..num_samples).map(|_| f32::round(repayment_hist_dist.sample(&mut rng))),
    );

    // Combine into features matrix
    let features = ndarray::stack![Axis(1), tx_count, wallet_age, avg_balance, repayment_hist];

    // Generate synthetic credit scores as a function of features with some noise
    let noise_dist = Normal::new(0.0, 0.05)?;
    let scores = features.map_axis(Axis(1), |row| {
        let score = 0.3 * row[0] + 0.2 * row[1] + 0.2 * row[2] + 0.3 * row[3];
        f32::min(f32::max(score + noise_dist.sample(&mut rng), 0.0), 1.0)
    });

    // Convert to Vec format for serialization
    let features_vec = features
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();

    let feature_names = vec![
        "tx_count".to_string(),
        "wallet_age".to_string(),
        "avg_balance".to_string(),
        "repayment_history".to_string(),
    ];

    Ok(CreditData {
        features: features_vec,
        scores: scores.to_vec(),
        feature_names,
        address_mapping: None,
    })
}

/// Calculate a credit score based on feature values using the same formula as in generate_synthetic_data
fn calculate_score(features: &[f32], noise: f32) -> f32 {
    let score = 0.3 * features[0] + 0.2 * features[1] + 0.2 * features[2] + 0.3 * features[3];
    f32::min(f32::max(score + noise, 0.0), 1.0)
}

/// Generates synthetic data with specific test addresses included
/// 
/// This function is similar to `generate_synthetic_data` but adds specific test addresses
/// with predetermined features to ensure they map to specific credit score tiers.
/// This is useful for testing the collateral tier system.
pub fn generate_synthetic_data_with_test_addresses(num_samples: usize) -> Result<CreditData> {
    // First generate random data
    let mut data = generate_synthetic_data(num_samples)?;
    
    // Create address mapping
    let mut address_mapping = HashMap::new();
    
    // Test address features - format: [tx_count, wallet_age, avg_balance, repayment_history]
    // Each feature is normalized to 0-1 range
    
    // Define test addresses for different tiers
    let low_tier_address = "0x2222222222222222222222222222222222222222".to_string();
    let medium_tier_address = "0x3333333333333333333333333333333333333333".to_string();
    let high_tier_address = "0x4444444444444444444444444444444444444444".to_string();
    let another_high_tier_address = "0x5555555555555555555555555555555555555555".to_string();
    
    // LOW tier address (score < 0.4)
    let low_tier_features = vec![0.1, 0.2, 0.1, 0.0];
    
    // MEDIUM tier address (score between 0.4 and 0.7)
    let medium_tier_features = vec![0.5, 0.4, 0.5, 1.0];
    
    // HIGH tier address (score > 0.7)
    let high_tier_features = vec![0.9, 0.8, 0.7, 1.0];
    
    // Another HIGH tier address
    let another_high_tier_features = vec![0.85, 0.9, 0.8, 1.0];
    
    // Calculate scores for each test address using the same formula as in the main function
    let low_tier_score = calculate_score(&low_tier_features, 0.0); // No noise for test addresses
    let medium_tier_score = calculate_score(&medium_tier_features, 0.0);
    let high_tier_score = calculate_score(&high_tier_features, 0.0);
    let another_high_tier_score = calculate_score(&another_high_tier_features, 0.0);
    
    // Add test addresses to the data
    let low_index = data.features.len();
    data.features.push(low_tier_features);
    data.scores.push(low_tier_score);
    address_mapping.insert(low_tier_address, low_index);
    
    let medium_index = data.features.len();
    data.features.push(medium_tier_features);
    data.scores.push(medium_tier_score);
    address_mapping.insert(medium_tier_address, medium_index);
    
    let high_index = data.features.len();
    data.features.push(high_tier_features);
    data.scores.push(high_tier_score);
    address_mapping.insert(high_tier_address, high_index);
    
    let another_high_index = data.features.len();
    data.features.push(another_high_tier_features);
    data.scores.push(another_high_tier_score);
    address_mapping.insert(another_high_tier_address, another_high_index);
    
    // Add address mapping to the data
    data.address_mapping = Some(address_mapping);
    
    Ok(data)
}

/// Add a specific Ethereum address to the dataset with custom features
/// 
/// This allows adding any number of addresses to the dataset with specific
/// features that will result in desired credit scores.
pub fn add_address_to_data(
    data: &mut CreditData, 
    address: &str, 
    features: Vec<f32>
) -> Result<()> {
    // Ensure we have an address mapping
    if data.address_mapping.is_none() {
        data.address_mapping = Some(HashMap::new());
    }
    
    // Calculate score for the features
    let score = calculate_score(&features, 0.0);
    
    // Add to the dataset
    let index = data.features.len();
    data.features.push(features);
    data.scores.push(score);
    
    // Update address mapping
    if let Some(ref mut mapping) = data.address_mapping {
        mapping.insert(address.to_string(), index);
    }
    
    Ok(())
}

pub fn save_data_as_json(data: &CreditData, path: &str) -> Result<()> {
    let json_data = serde_json::to_string_pretty(data)?;
    let mut file = File::create(path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}
