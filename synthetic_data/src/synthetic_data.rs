use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Serialize, Deserialize)]
pub struct CreditData {
    pub features: Vec<Vec<f32>>,
    pub scores: Vec<f32>,
    pub feature_names: Vec<String>,
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
    })
}

pub fn save_data_as_json(data: &CreditData, path: &str) -> Result<()> {
    let json_data = serde_json::to_string_pretty(data)?;
    let mut file = File::create(path)?;
    file.write_all(json_data.as_bytes())?;
    Ok(())
}
