package com.frauddetection.model;

import com.frauddetection.entity.Transaction;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

/**
 * Handles fraud prediction using trained models
 */
public class FraudPredictor {
    
    private List<Classifier> models;
    private List<String> modelNames;
    private Instances templateDataset;
    
    public FraudPredictor(List<Classifier> models, List<String> modelNames, Instances templateDataset) {
        this.models = models;
        this.modelNames = modelNames;
        this.templateDataset = templateDataset;
    }
    
    /**
     * Predict fraud for a single transaction
     * @param transaction Transaction to predict
     * @return PredictionResult with fraud probability and prediction
     */
    public PredictionResult predictFraud(Transaction transaction) {
        return predictFraud(transaction.getAllFeatures());
    }
    
    /**
     * Predict fraud using feature array
     * @param features Array of 30 features (time, amount, V1-V28)
     * @return PredictionResult with fraud probability and prediction
     */
    public PredictionResult predictFraud(double[] features) {
        if (features.length != 30) {
            throw new IllegalArgumentException("Expected 30 features, got " + features.length);
        }
        
        try {
            // Create instance for prediction
            Instance instance = new DenseInstance(31); // 30 features + 1 class
            instance.setDataset(templateDataset);
            
            for (int i = 0; i < 30; i++) {
                instance.setValue(i, features[i]);
            }
            instance.setValue(30, "0"); // Dummy class value
            
            // Get predictions from all models
            List<Double> fraudProbabilities = new java.util.ArrayList<>();
            List<Boolean> predictions = new java.util.ArrayList<>();
            
            for (int i = 0; i < models.size(); i++) {
                Classifier model = models.get(i);
                double[] distribution = model.distributionForInstance(instance);
                double fraudProbability = distribution[1]; // Probability of fraud class
                boolean isFraud = fraudProbability > 0.5;
                
                fraudProbabilities.add(fraudProbability);
                predictions.add(isFraud);
            }
            
            // Calculate ensemble prediction
            double avgFraudProbability = fraudProbabilities.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);
            
            boolean ensemblePrediction = avgFraudProbability > 0.5;
            
            return new PredictionResult(
                ensemblePrediction,
                avgFraudProbability,
                fraudProbabilities,
                predictions,
                modelNames
            );
            
        } catch (Exception e) {
            System.err.println("Error making prediction: " + e.getMessage());
            return new PredictionResult(false, 0.0, null, null, null);
        }
    }
    
    /**
     * Predict fraud using only V1-V28 features (without time and amount)
     * @param vFeatures Array of 28 V features
     * @return PredictionResult with fraud probability and prediction
     */
    public PredictionResult predictFraudVFeatures(double[] vFeatures) {
        if (vFeatures.length != 28) {
            throw new IllegalArgumentException("Expected 28 V features, got " + vFeatures.length);
        }
        
        // Create full feature array with dummy time and amount
        double[] fullFeatures = new double[30];
        fullFeatures[0] = 0.0; // Dummy time
        fullFeatures[1] = 0.0; // Dummy amount
        System.arraycopy(vFeatures, 0, fullFeatures, 2, 28);
        
        return predictFraud(fullFeatures);
    }
    
    /**
     * Get prediction from a specific model
     * @param modelIndex Index of the model to use
     * @param features Array of 30 features
     * @return PredictionResult from the specified model
     */
    public PredictionResult predictWithSpecificModel(int modelIndex, double[] features) {
        if (modelIndex < 0 || modelIndex >= models.size()) {
            throw new IllegalArgumentException("Invalid model index: " + modelIndex);
        }
        
        try {
            Instance instance = new DenseInstance(31);
            instance.setDataset(templateDataset);
            
            for (int i = 0; i < 30; i++) {
                instance.setValue(i, features[i]);
            }
            instance.setValue(30, "0");
            
            Classifier model = models.get(modelIndex);
            double[] distribution = model.distributionForInstance(instance);
            double fraudProbability = distribution[1];
            boolean isFraud = fraudProbability > 0.5;
            
            return new PredictionResult(
                isFraud,
                fraudProbability,
                List.of(fraudProbability),
                List.of(isFraud),
                List.of(modelNames.get(modelIndex))
            );
            
        } catch (Exception e) {
            System.err.println("Error making prediction with model " + modelIndex + ": " + e.getMessage());
            return new PredictionResult(false, 0.0, null, null, null);
        }
    }
    
    /**
     * PredictionResult class to hold prediction results
     */
    public static class PredictionResult {
        private boolean isFraud;
        private double fraudProbability;
        private List<Double> individualProbabilities;
        private List<Boolean> individualPredictions;
        private List<String> modelNames;
        
        public PredictionResult(boolean isFraud, double fraudProbability,
                              List<Double> individualProbabilities,
                              List<Boolean> individualPredictions,
                              List<String> modelNames) {
            this.isFraud = isFraud;
            this.fraudProbability = fraudProbability;
            this.individualProbabilities = individualProbabilities;
            this.individualPredictions = individualPredictions;
            this.modelNames = modelNames;
        }
        
        // Getters
        public boolean isFraud() { return isFraud; }
        public double getFraudProbability() { return fraudProbability; }
        public List<Double> getIndividualProbabilities() { return individualProbabilities; }
        public List<Boolean> getIndividualPredictions() { return individualPredictions; }
        public List<String> getModelNames() { return modelNames; }
        
        public String getRiskLevel() {
            if (fraudProbability < 0.3) return "LOW";
            else if (fraudProbability < 0.7) return "MEDIUM";
            else return "HIGH";
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Fraud Prediction: ").append(isFraud ? "FRAUD" : "NORMAL").append("\n");
            sb.append("Fraud Probability: ").append(String.format("%.4f", fraudProbability)).append("\n");
            sb.append("Risk Level: ").append(getRiskLevel()).append("\n");
            
            if (individualProbabilities != null && individualPredictions != null) {
                sb.append("Individual Model Results:\n");
                for (int i = 0; i < individualProbabilities.size(); i++) {
                    sb.append("  ").append(modelNames.get(i)).append(": ")
                      .append(individualPredictions.get(i) ? "FRAUD" : "NORMAL")
                      .append(" (").append(String.format("%.4f", individualProbabilities.get(i))).append(")\n");
                }
            }
            
            return sb.toString();
        }
    }
}
