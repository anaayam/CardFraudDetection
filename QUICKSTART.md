# Quick Start Guide

## Prerequisites

- Java 11 or higher
- Maven 3.6 or higher

## Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   mvn clean install
   ```

3. **Download the dataset (optional):**
   - Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the `data/` directory

## Running the Application

### Option 1: Using the run script (Recommended)
```bash
# On Unix/Linux/Mac
./run.sh

# On Windows
run.bat
```

### Option 2: Using Maven directly
```bash
# Compile and run
mvn clean compile exec:java

# Or run the demo (works without dataset)
mvn exec:java -Dexec.mainClass="com.frauddetection.DemoApp"
```

### Option 3: Build and run JAR
```bash
mvn package
java -jar target/card-fraud-detection-1.0.0.jar
```

## What the Application Does

1. **Loads the dataset** (or runs with sample data if not available)
2. **Preprocesses the data** (handles imbalanced classes)
3. **Trains multiple ML models:**
   - Random Forest
   - J48 Decision Tree
   - Naive Bayes
   - Support Vector Machine
   - Logistic Regression
4. **Evaluates model performance** with metrics like precision, recall, F1-score, and AUC
5. **Provides interactive prediction** for new transactions

## Sample Usage

The application will show you:
- Dataset statistics
- Model training progress
- Performance comparison of all models
- Interactive prediction interface

## Expected Output

```
=== Credit Card Fraud Detection System ===
Loading and preprocessing data...

=== Dataset Statistics ===
Total instances: 284807
Total attributes: 31
Class distribution:
  Normal: 284315 (99.83%)
  Fraud: 492 (0.17%)

Training multiple models...
Training Random Forest...
Training J48 Decision Tree...
...

=== Model Comparison ===
Model                    Accuracy   Precision  Recall     F1-Score   AUC
Random Forest           0.9995     0.8542     0.7895     0.8203     0.9234
J48 Decision Tree       0.9993     0.8123     0.7234     0.7654     0.9012
...

Best model: Random Forest (F1-Score: 0.8203)
```

## Troubleshooting

### Common Issues

1. **"Data file not found"**
   - Download the dataset from Kaggle
   - Place `creditcard.csv` in the `data/` directory
   - Or run the demo version without the dataset

2. **"Java not found"**
   - Install Java 11 or higher
   - Make sure Java is in your PATH

3. **"Maven not found"**
   - Install Maven 3.6 or higher
   - Make sure Maven is in your PATH

4. **Out of memory errors**
   - Increase JVM heap size: `export MAVEN_OPTS="-Xmx2g"`
   - Or use a smaller dataset subset

## Next Steps

- Experiment with different model parameters
- Try different feature engineering techniques
- Implement real-time fraud detection
- Add more sophisticated ensemble methods
