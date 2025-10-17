"""
Create sample data for testing the fraud detection system
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create a small sample dataset for testing"""
    print("Creating sample fraud detection dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000  # Small dataset for testing
    n_fraud = 20      # 2% fraud rate
    
    # Generate normal transactions
    normal_data = {
        'Time': np.random.uniform(0, 172792, n_samples - n_fraud),
        'Amount': np.random.exponential(100, n_samples - n_fraud),
        'Class': [0] * (n_samples - n_fraud)
    }
    
    # Generate fraud transactions
    fraud_data = {
        'Time': np.random.uniform(0, 172792, n_fraud),
        'Amount': np.random.exponential(50, n_fraud),  # Fraud transactions tend to be smaller
        'Class': [1] * n_fraud
    }
    
    # Generate V1-V28 features (PCA components)
    for i in range(1, 29):
        # Normal transactions: centered around 0
        normal_data[f'V{i}'] = np.random.normal(0, 1, n_samples - n_fraud)
        # Fraud transactions: different distribution
        fraud_data[f'V{i}'] = np.random.normal(0.5, 1.5, n_fraud)
    
    # Combine data
    normal_df = pd.DataFrame(normal_data)
    fraud_df = pd.DataFrame(fraud_data)
    
    # Combine and shuffle
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/creditcard.csv', index=False)
    
    print(f"✅ Sample dataset created with {len(df)} transactions")
    print(f"📊 Normal transactions: {len(df[df['Class'] == 0])}")
    print(f"🚨 Fraud transactions: {len(df[df['Class'] == 1])}")
    print(f"📈 Fraud rate: {len(df[df['Class'] == 1]) / len(df) * 100:.2f}%")
    print(f"💾 Saved to: data/creditcard.csv")
    
    return df

if __name__ == "__main__":
    create_sample_data()
