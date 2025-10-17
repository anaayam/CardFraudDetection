package com.frauddetection;

import com.frauddetection.model.DataProcessor;
import com.frauddetection.model.ModelTrainer;
import com.frauddetection.model.FraudPredictor;
import com.frauddetection.util.MetricsCalculator;
import weka.core.Instances;

import java.io.File;
import java.util.List;
import java.util.Scanner;

/**
 * Main application for Credit Card Fraud Detection
 */
public class FraudDetectionApp {
    
    private static final String DATA_FILE = "data/creditcard.csv";
    
    public static void main(String[] args) {
        System.out.println("=== Credit Card Fraud Detection System ===");
        System.out.println("Loading and preprocessing data...\n");
        
        try {
            // Check if data file exists
            File dataFile = new File(DATA_FILE);
            if (!dataFile.exists()) {
                System.out.println("ERROR: Data file not found at " + DATA_FILE);
                System.out.println("Please download the Credit Card Fraud Detection dataset from Kaggle");
                System.out.println("and place the 'creditcard.csv' file in the 'data/' directory.");
                return;
            }
            
            // Load and preprocess data
            DataProcessor processor = new DataProcessor();
            Instances dataset = processor.loadAndPreprocessData(DATA_FILE);
            processor.printDatasetStatistics(dataset);
            
            // Split data into train and test sets
            Instances[] trainTestSplit = processor.trainTestSplit(dataset, 0.2);
            Instances trainData = trainTestSplit[0];
            Instances testData = trainTestSplit[1];
            
            System.out.println("\nTraining set size: " + trainData.numInstances());
            System.out.println("Test set size: " + testData.numInstances());
            
            // Balance the training dataset
            System.out.println("\nBalancing training dataset...");
            Instances balancedTrainData = processor.balanceDataset(trainData);
            processor.printDatasetStatistics(balancedTrainData);
            
            // Train models
            System.out.println("\nTraining machine learning models...");
            ModelTrainer trainer = new ModelTrainer();
            List<weka.classifiers.Classifier> models = trainer.trainModels(balancedTrainData);
            List<String> modelNames = trainer.getModelNames();
            
            // Evaluate models
            System.out.println("\nEvaluating models...");
            MetricsCalculator metricsCalculator = new MetricsCalculator();
            List<MetricsCalculator.ModelMetrics> results = 
                metricsCalculator.evaluateModels(models, testData, modelNames);
            
            // Print results
            for (int i = 0; i < results.size(); i++) {
                metricsCalculator.printDetailedResults(results.get(i));
            }
            
            // Print comparison table
            metricsCalculator.printComparisonTable(results);
            
            // Find best model
            MetricsCalculator.ModelMetrics bestModel = metricsCalculator.findBestModel(results);
            System.out.println("\nBest model: " + bestModel.getModelName() + 
                             " (F1-Score: " + String.format("%.4f", bestModel.getF1ScoreFraud()) + ")");
            
            // Create predictor
            FraudPredictor predictor = new FraudPredictor(models, modelNames, dataset);
            
            // Interactive prediction demo
            runInteractiveDemo(predictor);
            
        } catch (Exception e) {
            System.err.println("Error in main application: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Run interactive prediction demo
     */
    private static void runInteractiveDemo(FraudPredictor predictor) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("\n=== Interactive Fraud Detection Demo ===");
        System.out.println("Enter transaction features to predict fraud (or 'quit' to exit)");
        System.out.println("Format: time,amount,v1,v2,v3,...,v28 (30 values total)");
        System.out.println("Example: 0,149.62,-0.073403,0.238,-0.073403,0.238,...");
        
        while (true) {
            System.out.print("\nEnter features (or 'quit'): ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("quit")) {
                break;
            }
            
            try {
                String[] values = input.split(",");
                if (values.length != 30) {
                    System.out.println("Error: Expected 30 features, got " + values.length);
                    continue;
                }
                
                double[] features = new double[30];
                for (int i = 0; i < 30; i++) {
                    features[i] = Double.parseDouble(values[i]);
                }
                
                FraudPredictor.PredictionResult result = predictor.predictFraud(features);
                System.out.println("\n" + result.toString());
                
            } catch (NumberFormatException e) {
                System.out.println("Error: Invalid number format. Please enter valid numbers.");
            } catch (Exception e) {
                System.out.println("Error: " + e.getMessage());
            }
        }
        
        scanner.close();
        System.out.println("Demo completed. Goodbye!");
    }
    
    /**
     * Run batch prediction demo with sample data
     */
    public static void runBatchDemo(FraudPredictor predictor) {
        System.out.println("\n=== Batch Prediction Demo ===");
        
        // Sample transactions (normalized features)
        double[][] sampleTransactions = {
            // Normal transaction
            {0, 149.62, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238,
             -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238,
             -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238, -0.073403, 0.238},
            
            // Potentially fraudulent transaction
            {0, 2.69, -1.359807, -0.072781, 2.536347, 1.378155, -0.338262, 0.462388, 0.239599, 0.098698,
             0.363787, 0.090794, -0.551601, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791,
             0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053}
        };
        
        String[] descriptions = {"Normal Transaction", "Suspicious Transaction"};
        
        for (int i = 0; i < sampleTransactions.length; i++) {
            System.out.println("\n" + descriptions[i] + ":");
            FraudPredictor.PredictionResult result = predictor.predictFraud(sampleTransactions[i]);
            System.out.println(result.toString());
        }
    }
}
