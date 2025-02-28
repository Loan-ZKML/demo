# Collateral Tier System Implementation Guide

This document explains how the collateral tier system works within our ZKML credit scoring system, along with implementation details and strategies for testing.

## Overview

DeFi lending protocols typically require substantial collateral (often 150% or more) to secure loans, making borrowing less capital-efficient. Our system introduces a privacy-preserving credit scoring mechanism that allows users to prove their creditworthiness without revealing sensitive financial data, resulting in reduced collateral requirements.

## Collateral Tier Structure

The system implements four distinct collateral tiers based on credit scores:

| Tier | Credit Score Range | Collateral Requirement | Description |
|------|-------------------|------------------------|-------------|
| UNKNOWN | No score/proof | 120% | Default tier for new users with no verifiable credit history |
| LOW | < 0.4 | 100% | Users with minimal credit history or lower scores |
| MEDIUM | 0.4 - 0.7 | 90% | Users with moderate credit history and scores |
| HIGH | > 0.7 | 80% | Users with excellent credit history and scores |

## Ethereum Address Strategy for Testing

### Test Address Convention

For testing and development purposes, we use a consistent set of Ethereum addresses to represent users in different tiers:

- **UNKNOWN Tier**: Any address not included in training data, e.g., `0x1111111111111111111111111111111111111111`
- **LOW Tier**: `0x2222222222222222222222222222222222222222`
- **MEDIUM Tier**: `0x3333333333333333333333333333333333333333`
- **HIGH Tier**: `0x4444444444444444444444444444444444444444`

### Training Data Integration

When training the model, include the test addresses with appropriate feature values that would result in the desired credit scores:

```rust
// Example feature values for test addresses (pseudo-code)
// Format: [tx_count, wallet_age, avg_balance, repayment_history]

// LOW tier address (score < 0.4)
data.add_record("0x2222222222222222222222222222222222222222", [0.1, 0.2, 0.1, 0.0]);

// MEDIUM tier address (score between 0.4 and 0.7)
data.add_record("0x3333333333333333333333333333333333333333", [0.5, 0.4, 0.5, 1.0]);

// HIGH tier address (score > 0.7)
data.add_record("0x4444444444444444444444444444444444444444", [0.9, 0.8, 0.7, 1.0]);
```

### Implementation in Smart Contracts

The on-chain collateral calculator would implement logic similar to:

```solidity
contract CollateralCalculator is ICollateralCalculator {
    mapping(address => CreditTier) private addressTiers;
    mapping(CreditTier => uint256) private tierCollateralPercentages;
    
    constructor() {
        // Set default collateral percentages for each tier (in basis points)
        tierCollateralPercentages[CreditTier.UNKNOWN] = 12000; // 120%
        tierCollateralPercentages[CreditTier.LOW] = 10000;     // 100%
        tierCollateralPercentages[CreditTier.MEDIUM] = 9000;   //  90%
        tierCollateralPercentages[CreditTier.HIGH] = 8000;     //  80%
    }
    
    function getCollateralRequirement(address _borrower, uint256 _loanAmount) 
        external 
        view 
        override 
        returns (CollateralRequirement memory) 
    {
        // Get the borrower's tier, defaulting to UNKNOWN
        CreditTier tier = addressTiers[_borrower];
        
        // Get the collateral percentage for this tier
        uint256 percentage = tierCollateralPercentages[tier];
        
        // Calculate required collateral amount
        uint256 requiredAmount = (_loanAmount * percentage) / 10000;
        
        return CollateralRequirement({
            requiredPercentage: percentage,
            requiredAmount: requiredAmount,
            tier: tier
        });
    }
    
    function updateCreditScore(
        address _borrower, 
        uint256 _creditScore, 
        bool _proofValidated
    ) external override returns (CreditTier) {
        require(_proofValidated, "Invalid proof");
        
        // Determine tier based on credit score
        CreditTier newTier;
        if (_creditScore < 400) {         // < 0.4
            newTier = CreditTier.LOW;
        } else if (_creditScore < 700) {  // 0.4 - 0.7
            newTier = CreditTier.MEDIUM;
        } else {                          // > 0.7
            newTier = CreditTier.HIGH;
        }
        
        // Update the borrower's tier
        addressTiers[_borrower] = newTier;
        
        return newTier;
    }
}
```

## Testing Strategy

### End-to-End Test Flow

1. **Generate Synthetic Data**: Create synthetic data that includes test addresses with features corresponding to their desired tiers.

2. **Train Model**: Train the model on this data to ensure the test addresses receive appropriate credit scores.

3. **Generate Proofs**: Create ZK proofs for each test address:
   - For the HIGH tier address, create a proof of a credit score > 0.7
   - For the MEDIUM tier address, create a proof of a credit score between 0.4 and 0.7
   - For the LOW tier address, create a proof of a credit score < 0.4
   - For the UNKNOWN tier test, use an address not in the training data

4. **Verify Collateral Calculation**: Test the smart contract's collateral calculations for each test address.

### Unknown User Testing

For testing the UNKNOWN tier (users without proof or history):

1. Use an Ethereum address not included in the training data
2. Attempt to request a loan without providing a ZK proof
3. Verify that the contract requires 120% collateral

### Key Test Cases

- **New user with no proof**: Should receive 120% collateral requirement
- **User with LOW score proof**: Should receive 100% collateral requirement
- **User with MEDIUM score proof**: Should receive 90% collateral requirement
- **User with HIGH score proof**: Should receive 80% collateral requirement
- **User with expired/invalid proof**: Should revert to 120% collateral requirement
- **Loan request with insufficient collateral**: Should be rejected

## Implementation Considerations

- **Proof Expiration**: Consider implementing a time-based expiration for proofs (e.g., 30 days) to ensure users maintain good standing.
- **Tier Transitions**: Monitor and handle cases where users move between tiers.
- **Interest Rate Variation**: Consider varying interest rates based on tiers in addition to collateral requirements.
- **Governance**: Consider implementing a governance mechanism for adjusting tier thresholds and collateral percentages.

## Integration with Frontend/CLI

The frontend or CLI application should follow these steps to interact with the system:

1. Generate credit score proof using the user's Ethereum address and financial data
2. Submit the proof to the smart contract
3. Request collateral requirement preview based on the verified proof
4. Submit loan request with appropriate collateral