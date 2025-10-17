# Data Directory

This directory should contain the Credit Card Fraud Detection dataset.

## Required File

- `creditcard.csv` - The main dataset file from Kaggle

## Dataset Information

The Credit Card Fraud Detection dataset contains:

- **Total transactions**: 284,807
- **Fraudulent transactions**: 492 (0.172%)
- **Features**: 30 (Time, Amount, V1-V28)
- **V1-V28**: PCA-transformed features for privacy
- **Class**: 0 (Normal) or 1 (Fraud)

## Download Instructions

1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download the `creditcard.csv` file
3. Place it in this directory (`data/creditcard.csv`)

## File Format

The CSV file should have the following columns:
```
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
```

## Privacy Note

The original features have been transformed using PCA (Principal Component Analysis) to protect the privacy of the cardholders. The V1-V28 features are the result of this transformation.
