package com.frauddetection;

import com.frauddetection.model.DataProcessor;
import com.frauddetection.model.ModelTrainer;
import com.frauddetection.model.FraudPredictor;
import com.frauddetection.util.MetricsCalculator;
import weka.core.Instances;

import java.io.File;
import java.util.List;

/**
 * Demo application for Credit Card Fraud Detection
 * This is a simplified version for demonstration purposes
 */
public class DemoApp {
    
    private static final String DATA_FILE = "data/creditcard.csv";
    
    public static void main(String[] args) {
        System.out.println("=== Credit Card Fraud Detection Demo ===");
        
        try {
            // Check if data file exists
            File dataFile = new File(DATA_FILE);
            if (!dataFile.exists()) {
                System.out.println("ERROR: Data file not found at " + DATA_FILE);
                System.out.println("Please download the Credit Card Fraud Detection dataset from Kaggle");
                System.out.println("and place the 'creditcard.csv' file in the 'data/' directory.");
                System.out.println("\nFor demo purposes, we'll use sample data...");
                runSampleDemo();
                return;
            }
            
            // Load and preprocess data
            System.out.println("Loading dataset...");
            DataProcessor processor = new DataProcessor();
            Instances dataset = processor.loadAndPreprocessData(DATA_FILE);
            
            // Use a smaller subset for demo
            Instances demoDataset = new Instances(dataset, 0, Math.min(10000, dataset.numInstances()));
            System.out.println("Using " + demoDataset.numInstances() + " instances for demo");
            
            // Split data
            Instances[] trainTestSplit = processor.trainTestSplit(demoDataset, 0.3);
            Instances trainData = trainTestSplit[0];
            Instances testData = trainTestSplit[1];
            
            // Balance training data
            Instances balancedTrainData = processor.balanceDataset(trainData);
            
            // Train models
            System.out.println("Training models...");
            ModelTrainer trainer = new ModelTrainer();
            List<weka.classifiers.Classifier> models = trainer.trainModels(balancedTrainData);
            List<String> modelNames = trainer.getModelNames();
            
            // Evaluate models
            System.out.println("Evaluating models...");
            MetricsCalculator metricsCalculator = new MetricsCalculator();
            List<MetricsCalculator.ModelMetrics> results = 
                metricsCalculator.evaluateModels(models, testData, modelNames);
            
            // Print results
            metricsCalculator.printComparisonTable(results);
            
            // Find best model
            MetricsCalculator.ModelMetrics bestModel = metricsCalculator.findBestModel(results);
            System.out.println("\nBest model: " + bestModel.getModelName());
            
            // Create predictor and run demo
            FraudPredictor predictor = new FraudPredictor(models, modelNames, dataset);
            FraudDetectionApp.runBatchDemo(predictor);
            
        } catch (Exception e) {
            System.err.println("Error in demo: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Run demo with sample data when dataset is not available
     */
    private static void runSampleDemo() {
        System.out.println("\n=== Sample Data Demo ===");
        System.out.println("This demo shows how the system would work with sample transaction data.");
        
        // Sample transaction features (simplified)
        double[] normalTransaction = {
            0, 149.62, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238,
            -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238,
            -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238
        };
        
        double[] suspiciousTransaction = {
            0, 2.69, -1.359807, -0.072781, 2.536347, 1.378155, -0.338262, 0.462388, 0.239599, 0.098698,
            0.363787, 0.090794, -0.551601, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791,
            0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053
        };
        
        System.out.println("\nSample Normal Transaction:");
        System.out.println("Features: Time=0, Amount=149.62, V1-V28 features...");
        System.out.println("Expected: NORMAL (Low fraud probability)");
        
        System.out.println("\nSample Suspicious Transaction:");
        System.out.println("Features: Time=0, Amount=2.69, V1-V28 features...");
        System.out.println("Expected: FRAUD (High fraud probability)");
        
        System.out.println("\nNote: To run the full demo with actual predictions,");
        System.out.println("please download the dataset from Kaggle and place it in the data/ directory.");
    }
}
