#!/bin/bash

# Credit Card Fraud Detection - Run Script

echo "=== Credit Card Fraud Detection System ==="

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed or not in PATH"
    echo "Please install Java 11 or higher"
    exit 1
fi

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Error: Maven is not installed or not in PATH"
    echo "Please install Maven 3.6 or higher"
    exit 1
fi

# Compile the project
echo "Compiling project..."
mvn clean compile

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

# Check if data file exists
if [ ! -f "data/creditcard.csv" ]; then
    echo "Warning: Data file not found at data/creditcard.csv"
    echo "Running demo with sample data..."
    mvn exec:java -Dexec.mainClass="com.frauddetection.DemoApp"
else
    echo "Running full application..."
    mvn exec:java -Dexec.mainClass="com.frauddetection.FraudDetectionApp"
fi
