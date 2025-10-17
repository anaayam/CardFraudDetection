package com.frauddetection.model;

import com.frauddetection.entity.Transaction;
import com.frauddetection.util.DataUtils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Handles data loading, preprocessing, and conversion to Weka format
 */
public class DataProcessor {
    
    private static final String CSV_HEADER = "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class";
    
    /**
     * Load and preprocess data from CSV file
     * @param csvFilePath Path to the CSV file
     * @return Weka Instances object
     */
    public Instances loadAndPreprocessData(String csvFilePath) throws IOException {
        List<Transaction> transactions = loadTransactionsFromCSV(csvFilePath);
        return convertToWekaInstances(transactions);
    }
    
    /**
     * Load transactions from CSV file
     * @param csvFilePath Path to the CSV file
     * @return List of Transaction objects
     */
    public List<Transaction> loadTransactionsFromCSV(String csvFilePath) throws IOException {
        List<Transaction> transactions = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFilePath))) {
            String line = reader.readLine(); // Skip header
            
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length >= 31) { // Time + 28 V features + Amount + Class
                    Transaction transaction = parseTransaction(values);
                    transactions.add(transaction);
                }
            }
        }
        
        System.out.println("Loaded " + transactions.size() + " transactions");
        return transactions;
    }
    
    /**
     * Parse a single transaction from CSV row
     * @param values Array of string values from CSV
     * @return Transaction object
     */
    private Transaction parseTransaction(String[] values) {
        double time = Double.parseDouble(values[0]);
        double amount = Double.parseDouble(values[29]);
        int classLabel = Integer.parseInt(values[30]);
        
        double[] vFeatures = new double[28];
        for (int i = 0; i < 28; i++) {
            vFeatures[i] = Double.parseDouble(values[i + 1]);
        }
        
        return new Transaction(time, amount, vFeatures, classLabel);
    }
    
    /**
     * Convert transactions to Weka Instances format
     * @param transactions List of Transaction objects
     * @return Weka Instances object
     */
    public Instances convertToWekaInstances(List<Transaction> transactions) {
        // Create attributes
        FastVector attributes = new FastVector();
        
        // Add feature attributes
        for (int i = 0; i < 30; i++) { // 30 features: time + amount + 28 V features
            attributes.addElement(new Attribute("feature_" + i));
        }
        
        // Add class attribute
        FastVector classValues = new FastVector();
        classValues.addElement("0"); // Normal
        classValues.addElement("1"); // Fraud
        attributes.addElement(new Attribute("class", classValues));
        
        // Create Instances object
        Instances dataset = new Instances("CreditCardFraud", attributes, transactions.size());
        dataset.setClassIndex(30); // Set class attribute index
        
        // Add instances
        for (Transaction transaction : transactions) {
            Instance instance = new DenseInstance(31); // 30 features + 1 class
            instance.setDataset(dataset);
            
            double[] features = transaction.getAllFeatures();
            for (int i = 0; i < 30; i++) {
                instance.setValue(i, features[i]);
            }
            instance.setValue(30, String.valueOf(transaction.getClassLabel()));
            
            dataset.add(instance);
        }
        
        System.out.println("Created Weka dataset with " + dataset.numInstances() + " instances and " + dataset.numAttributes() + " attributes");
        return dataset;
    }
    
    /**
     * Split dataset into training and testing sets
     * @param dataset Original dataset
     * @param testRatio Ratio of test data (0.0 to 1.0)
     * @return Array containing [trainDataset, testDataset]
     */
    public Instances[] trainTestSplit(Instances dataset, double testRatio) {
        dataset.randomize(new java.util.Random(42));
        
        int testSize = (int) (dataset.numInstances() * testRatio);
        int trainSize = dataset.numInstances() - testSize;
        
        Instances trainDataset = new Instances(dataset, 0, trainSize);
        Instances testDataset = new Instances(dataset, trainSize, testSize);
        
        return new Instances[]{trainDataset, testDataset};
    }
    
    /**
     * Balance dataset using undersampling for majority class
     * @param dataset Original dataset
     * @return Balanced dataset
     */
    public Instances balanceDataset(Instances dataset) {
        // Count classes
        int[] classCounts = new int[2];
        for (int i = 0; i < dataset.numInstances(); i++) {
            int classValue = (int) dataset.instance(i).classValue();
            classCounts[classValue]++;
        }
        
        System.out.println("Original class distribution: Normal=" + classCounts[0] + ", Fraud=" + classCounts[1]);
        
        // Find minority class size
        int minoritySize = Math.min(classCounts[0], classCounts[1]);
        
        // Create balanced dataset
        Instances balancedDataset = new Instances(dataset, 0);
        
        // Collect indices for each class
        List<Integer> normalIndices = new ArrayList<>();
        List<Integer> fraudIndices = new ArrayList<>();
        
        for (int i = 0; i < dataset.numInstances(); i++) {
            int classValue = (int) dataset.instance(i).classValue();
            if (classValue == 0) {
                normalIndices.add(i);
            } else {
                fraudIndices.add(i);
            }
        }
        
        // Shuffle indices
        Collections.shuffle(normalIndices, new Random(42));
        Collections.shuffle(fraudIndices, new Random(42));
        
        // Add balanced samples
        for (int i = 0; i < minoritySize; i++) {
            balancedDataset.add(dataset.instance(normalIndices.get(i)));
            balancedDataset.add(dataset.instance(fraudIndices.get(i)));
        }
        
        System.out.println("Balanced dataset size: " + balancedDataset.numInstances());
        return balancedDataset;
    }
    
    /**
     * Get class distribution statistics
     * @param dataset Dataset to analyze
     * @return Map with class distribution
     */
    public Map<String, Integer> getClassDistribution(Instances dataset) {
        Map<String, Integer> distribution = new HashMap<>();
        distribution.put("Normal", 0);
        distribution.put("Fraud", 0);
        
        for (int i = 0; i < dataset.numInstances(); i++) {
            int classValue = (int) dataset.instance(i).classValue();
            if (classValue == 0) {
                distribution.put("Normal", distribution.get("Normal") + 1);
            } else {
                distribution.put("Fraud", distribution.get("Fraud") + 1);
            }
        }
        
        return distribution;
    }
    
    /**
     * Print dataset statistics
     * @param dataset Dataset to analyze
     */
    public void printDatasetStatistics(Instances dataset) {
        System.out.println("\n=== Dataset Statistics ===");
        System.out.println("Total instances: " + dataset.numInstances());
        System.out.println("Total attributes: " + dataset.numAttributes());
        
        Map<String, Integer> distribution = getClassDistribution(dataset);
        System.out.println("Class distribution:");
        for (Map.Entry<String, Integer> entry : distribution.entrySet()) {
            double percentage = (double) entry.getValue() / dataset.numInstances() * 100;
            System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " (" + String.format("%.2f", percentage) + "%)");
        }
    }
}
