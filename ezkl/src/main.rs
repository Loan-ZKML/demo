mod direct_ezkl;
mod onnx_converter;

use anyhow::{Result, Context};
use std::path::Path;
use std::process::Command;
use direct_ezkl::run_ezkl_process;

use synthetic_data::{generate_synthetic_data_with_test_addresses, save_data_as_json};
use synthetic_data::model_trainer::CreditScoreModel;

fn main() -> Result<()> {
    // Step 1: Generate synthetic data with test addresses
    println!("Generating synthetic credit data with test addresses...");
    let data = generate_synthetic_data_with_test_addresses(1000)?;
    save_data_as_json(&data, "credit_data.json")?;
    println!("Saved synthetic data to credit_data.json");
    
    // Log information about the test addresses
    println!("Included test addresses for different collateral tiers:");
    println!("  LOW tier (100% collateral): 0x2222222222222222222222222222222222222222");
    println!("  MEDIUM tier (90% collateral): 0x276ef71c8F12508d187E7D8Fcc2FE6A38a5884B1");
    println!("  HIGH tier (80% collateral): 0x4444444444444444444444444444444444444444");

    // Step 2: Train model
    println!("Training credit score model...");
    let model = CreditScoreModel::train(&data)?;

    // Step 3: Export model to JSON
    println!("Exporting model to JSON format...");
    model.export_to_json(Path::new("credit_model.json"))?;

    // Step 4: Save sample input for EZKL
    println!("Saving sample input for EZKL...");
    model.save_sample_input(&data, "input.json")?;

    // Step 5: Process with EZKL using the shell script
    println!("\nStarting EZKL processing with the real EZKL binary...");
    
    // Try to find the script in the project directory structure
    // First, check in the current directory
    let mut script_path = Path::new("run_ezkl.sh").to_path_buf();
    
    // If not found, check in the ezkl subdirectory
    if !script_path.exists() {
        script_path = Path::new("ezkl/run_ezkl.sh").to_path_buf();
    }
    
    // If still not found, check relative to the crate root
    if !script_path.exists() {
        // Get the directory where this source file is located (ezkl/src)
        let crate_root = std::env::current_dir()
            .context("Failed to get current directory")?;
        script_path = crate_root.join("run_ezkl.sh");
    }
    
    // Final check
    if !script_path.exists() {
        return Err(anyhow::anyhow!(
            "EZKL script not found. Make sure run_ezkl.sh exists in the ezkl directory."
        ));
    }
    
    println!("Found script at: {}", script_path.display());
    
    // Run the shell script
    println!("Executing: {}", script_path.display());
    
    // Get the directory containing the script to set as the current working directory
    let script_dir = script_path.parent()
        .context("Failed to get script directory")?;
    
    // Convert the path to an absolute path to ensure it can be found
    let absolute_script_path = script_path.canonicalize()
        .context("Failed to get absolute path to script")?;
    
    println!("Absolute path: {}", absolute_script_path.display());
    
    // Try multiple approaches to run the script
    println!("Attempting to run script using multiple methods...");
    
    // Method 1: Use full path to bash and absolute script path
    let result1 = Command::new("/bin/bash")
        .arg(&absolute_script_path)
        .current_dir(script_dir)
        .status();
    
    if let Ok(status) = result1 {
        if status.success() {
            return Ok(());
        } else {
            println!("Method 1 failed with status: {}", status);
        }
    } else {
        println!("Method 1 failed: {:?}", result1.err());
    }
    
    // Method 2: Use sh instead of bash
    println!("Trying with /bin/sh...");
    let result2 = Command::new("/bin/sh")
        .arg(&absolute_script_path)
        .current_dir(script_dir)
        .status();
    
    if let Ok(status) = result2 {
        if status.success() {
            return Ok(());
        } else {
            println!("Method 2 failed with status: {}", status);
        }
    } else {
        println!("Method 2 failed: {:?}", result2.err());
    }
    
    // Method 3: Use system
    println!("Trying with system command...");
    
    let cmd = format!("cd {} && /bin/bash {}", 
        script_dir.display(),
        absolute_script_path.display());
    
    println!("Executing system command: {}", cmd);
    
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .status()
        .context("Failed to execute EZKL script with system command")?;
    
    if !status.success() {
        println!("All script execution methods failed. Falling back to direct EZKL implementation...");
        
        // Get the current working directory for the direct implementation
        let current_dir = std::env::current_dir()
            .context("Failed to get current directory")?;
        
        // Try running EZKL directly without using the script
        run_ezkl_process(&current_dir)?;
        
        println!("\nZKML credit scoring system setup completed using direct implementation!");
        return Ok(());
    }

    println!("\nZKML credit scoring system setup complete!");
    println!("All artifacts have been generated. Check the directory for:");
    println!(" - Model: credit_model.json");
    println!(" - Circuit: model.compiled");
    println!(" - Verification key: vk.key");
    println!(" - Proving key: pk.key");
    println!(" - Proof: proof.json");
    println!(" - EVM verifier contract: CreditVerifier.sol");

    Ok(())
}