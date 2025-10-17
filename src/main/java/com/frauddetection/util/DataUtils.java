package com.frauddetection.util;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.Collections;

/**
 * Utility class for data manipulation and statistical operations
 */
public class DataUtils {
    
    private static final Random random = new Random(42); // Fixed seed for reproducibility
    
    /**
     * Normalize features using min-max scaling
     * @param data List of feature arrays
     * @return Normalized data
     */
    public static List<double[]> normalizeFeatures(List<double[]> data) {
        if (data.isEmpty()) return data;
        
        int featureCount = data.get(0).length;
        double[] minValues = new double[featureCount];
        double[] maxValues = new double[featureCount];
        
        // Initialize with first row
        for (int i = 0; i < featureCount; i++) {
            minValues[i] = data.get(0)[i];
            maxValues[i] = data.get(0)[i];
        }
        
        // Find min and max values
        for (double[] row : data) {
            for (int i = 0; i < featureCount; i++) {
                minValues[i] = Math.min(minValues[i], row[i]);
                maxValues[i] = Math.max(maxValues[i], row[i]);
            }
        }
        
        // Normalize data
        List<double[]> normalizedData = new ArrayList<>();
        for (double[] row : data) {
            double[] normalizedRow = new double[featureCount];
            for (int i = 0; i < featureCount; i++) {
                if (maxValues[i] - minValues[i] != 0) {
                    normalizedRow[i] = (row[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                } else {
                    normalizedRow[i] = 0.5; // Default value for constant features
                }
            }
            normalizedData.add(normalizedRow);
        }
        
        return normalizedData;
    }
    
    /**
     * Standardize features using z-score normalization
     * @param data List of feature arrays
     * @return Standardized data
     */
    public static List<double[]> standardizeFeatures(List<double[]> data) {
        if (data.isEmpty()) return data;
        
        int featureCount = data.get(0).length;
        double[] means = new double[featureCount];
        double[] stdDevs = new double[featureCount];
        
        // Calculate means
        for (double[] row : data) {
            for (int i = 0; i < featureCount; i++) {
                means[i] += row[i];
            }
        }
        for (int i = 0; i < featureCount; i++) {
            means[i] /= data.size();
        }
        
        // Calculate standard deviations
        for (double[] row : data) {
            for (int i = 0; i < featureCount; i++) {
                stdDevs[i] += Math.pow(row[i] - means[i], 2);
            }
        }
        for (int i = 0; i < featureCount; i++) {
            stdDevs[i] = Math.sqrt(stdDevs[i] / data.size());
        }
        
        // Standardize data
        List<double[]> standardizedData = new ArrayList<>();
        for (double[] row : data) {
            double[] standardizedRow = new double[featureCount];
            for (int i = 0; i < featureCount; i++) {
                if (stdDevs[i] != 0) {
                    standardizedRow[i] = (row[i] - means[i]) / stdDevs[i];
                } else {
                    standardizedRow[i] = 0; // Default value for constant features
                }
            }
            standardizedData.add(standardizedRow);
        }
        
        return standardizedData;
    }
    
    /**
     * Split data into training and testing sets
     * @param data List of feature arrays
     * @param labels List of class labels
     * @param testRatio Ratio of test data (0.0 to 1.0)
     * @return Array containing [trainData, trainLabels, testData, testLabels]
     */
    @SuppressWarnings("unchecked")
    public static List<double[]>[] trainTestSplit(List<double[]> data, List<Integer> labels, double testRatio) {
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException("Data and labels must have the same size");
        }
        
        // Create indices and shuffle
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        int testSize = (int) (data.size() * testRatio);
        int trainSize = data.size() - testSize;
        
        List<double[]> trainData = new ArrayList<>();
        List<Integer> trainLabels = new ArrayList<>();
        List<double[]> testData = new ArrayList<>();
        List<Integer> testLabels = new ArrayList<>();
        
        // Split data
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            trainData.add(data.get(idx));
            trainLabels.add(labels.get(idx));
        }
        
        for (int i = trainSize; i < data.size(); i++) {
            int idx = indices.get(i);
            testData.add(data.get(idx));
            testLabels.add(labels.get(idx));
        }
        
        return new List[]{trainData, trainLabels, testData, testLabels};
    }
    
    /**
     * Calculate class distribution
     * @param labels List of class labels
     * @return Array with counts for each class
     */
    public static int[] getClassDistribution(List<Integer> labels) {
        int maxClass = Collections.max(labels);
        int[] distribution = new int[maxClass + 1];
        
        for (int label : labels) {
            distribution[label]++;
        }
        
        return distribution;
    }
    
    /**
     * Shuffle data and labels together
     * @param data List of feature arrays
     * @param labels List of class labels
     */
    public static void shuffleData(List<double[]> data, List<Integer> labels) {
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException("Data and labels must have the same size");
        }
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        List<double[]> shuffledData = new ArrayList<>();
        List<Integer> shuffledLabels = new ArrayList<>();
        
        for (int idx : indices) {
            shuffledData.add(data.get(idx));
            shuffledLabels.add(labels.get(idx));
        }
        
        data.clear();
        labels.clear();
        data.addAll(shuffledData);
        labels.addAll(shuffledLabels);
    }
}
