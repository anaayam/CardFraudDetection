package com.frauddetection.util;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;

import java.util.List;
import java.util.ArrayList;

/**
 * Calculates various performance metrics for model evaluation
 */
public class MetricsCalculator {
    
    /**
     * Evaluate a single model and return metrics
     * @param model Trained classifier
     * @param testData Test dataset
     * @return ModelMetrics object with performance metrics
     */
    public ModelMetrics evaluateModel(Classifier model, Instances testData) {
        try {
            Evaluation evaluation = new Evaluation(testData);
            evaluation.evaluateModel(model, testData);
            
            return new ModelMetrics(
                evaluation.correct(),
                evaluation.incorrect(),
                evaluation.numInstances(),
                evaluation.precision(1), // Precision for fraud class
                evaluation.recall(1),    // Recall for fraud class
                evaluation.fMeasure(1),  // F1-score for fraud class
                evaluation.areaUnderROC(1), // AUC for fraud class
                evaluation.precision(0), // Precision for normal class
                evaluation.recall(0),    // Recall for normal class
                evaluation.fMeasure(0)   // F1-score for normal class
            );
        } catch (Exception e) {
            System.err.println("Error evaluating model: " + e.getMessage());
            return new ModelMetrics();
        }
    }
    
    /**
     * Evaluate multiple models and return comparison
     * @param models List of trained classifiers
     * @param testData Test dataset
     * @param modelNames List of model names
     * @return List of ModelMetrics for each model
     */
    public List<ModelMetrics> evaluateModels(List<Classifier> models, Instances testData, List<String> modelNames) {
        List<ModelMetrics> results = new ArrayList<>();
        
        for (int i = 0; i < models.size(); i++) {
            System.out.println("Evaluating " + modelNames.get(i) + "...");
            ModelMetrics metrics = evaluateModel(models.get(i), testData);
            metrics.setModelName(modelNames.get(i));
            results.add(metrics);
        }
        
        return results;
    }
    
    /**
     * Calculate confusion matrix
     * @param model Trained classifier
     * @param testData Test dataset
     * @return 2x2 confusion matrix [TN, FP, FN, TP]
     */
    public int[] calculateConfusionMatrix(Classifier model, Instances testData) {
        int[] matrix = new int[4]; // [TN, FP, FN, TP]
        
        try {
            for (int i = 0; i < testData.numInstances(); i++) {
                Instance instance = testData.instance(i);
                double actualClass = instance.classValue();
                double predictedClass = model.classifyInstance(instance);
                
                if (actualClass == 0 && predictedClass == 0) {
                    matrix[0]++; // True Negative
                } else if (actualClass == 0 && predictedClass == 1) {
                    matrix[1]++; // False Positive
                } else if (actualClass == 1 && predictedClass == 0) {
                    matrix[2]++; // False Negative
                } else if (actualClass == 1 && predictedClass == 1) {
                    matrix[3]++; // True Positive
                }
            }
        } catch (Exception e) {
            System.err.println("Error calculating confusion matrix: " + e.getMessage());
        }
        
        return matrix;
    }
    
    /**
     * Print detailed evaluation results
     * @param metrics ModelMetrics object
     */
    public void printDetailedResults(ModelMetrics metrics) {
        System.out.println("\n=== " + metrics.getModelName() + " Results ===");
        System.out.println("Accuracy: " + String.format("%.4f", metrics.getAccuracy()));
        System.out.println("Precision (Fraud): " + String.format("%.4f", metrics.getPrecisionFraud()));
        System.out.println("Recall (Fraud): " + String.format("%.4f", metrics.getRecallFraud()));
        System.out.println("F1-Score (Fraud): " + String.format("%.4f", metrics.getF1ScoreFraud()));
        System.out.println("AUC: " + String.format("%.4f", metrics.getAuc()));
        System.out.println("Precision (Normal): " + String.format("%.4f", metrics.getPrecisionNormal()));
        System.out.println("Recall (Normal): " + String.format("%.4f", metrics.getRecallNormal()));
        System.out.println("F1-Score (Normal): " + String.format("%.4f", metrics.getF1ScoreNormal()));
    }
    
    /**
     * Print comparison table of all models
     * @param results List of ModelMetrics for all models
     */
    public void printComparisonTable(List<ModelMetrics> results) {
        System.out.println("\n=== Model Comparison ===");
        System.out.printf("%-25s %-10s %-10s %-10s %-10s %-10s%n", 
                        "Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC");
        System.out.println("=".repeat(80));
        
        for (ModelMetrics metrics : results) {
            System.out.printf("%-25s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%n",
                            metrics.getModelName(),
                            metrics.getAccuracy(),
                            metrics.getPrecisionFraud(),
                            metrics.getRecallFraud(),
                            metrics.getF1ScoreFraud(),
                            metrics.getAuc());
        }
    }
    
    /**
     * Find the best model based on F1-score
     * @param results List of ModelMetrics for all models
     * @return Best model metrics
     */
    public ModelMetrics findBestModel(List<ModelMetrics> results) {
        ModelMetrics best = results.get(0);
        double bestF1 = best.getF1ScoreFraud();
        
        for (ModelMetrics metrics : results) {
            if (metrics.getF1ScoreFraud() > bestF1) {
                best = metrics;
                bestF1 = metrics.getF1ScoreFraud();
            }
        }
        
        return best;
    }
    
    /**
     * ModelMetrics class to hold evaluation results
     */
    public static class ModelMetrics {
        private String modelName;
        private double correct;
        private double incorrect;
        private double totalInstances;
        private double precisionFraud;
        private double recallFraud;
        private double f1ScoreFraud;
        private double auc;
        private double precisionNormal;
        private double recallNormal;
        private double f1ScoreNormal;
        
        public ModelMetrics() {
            // Default constructor
        }
        
        public ModelMetrics(double correct, double incorrect, double totalInstances,
                          double precisionFraud, double recallFraud, double f1ScoreFraud, double auc,
                          double precisionNormal, double recallNormal, double f1ScoreNormal) {
            this.correct = correct;
            this.incorrect = incorrect;
            this.totalInstances = totalInstances;
            this.precisionFraud = precisionFraud;
            this.recallFraud = recallFraud;
            this.f1ScoreFraud = f1ScoreFraud;
            this.auc = auc;
            this.precisionNormal = precisionNormal;
            this.recallNormal = recallNormal;
            this.f1ScoreNormal = f1ScoreNormal;
        }
        
        // Getters and Setters
        public String getModelName() { return modelName; }
        public void setModelName(String modelName) { this.modelName = modelName; }
        
        public double getAccuracy() { return correct / totalInstances; }
        public double getCorrect() { return correct; }
        public double getIncorrect() { return incorrect; }
        public double getTotalInstances() { return totalInstances; }
        public double getPrecisionFraud() { return precisionFraud; }
        public double getRecallFraud() { return recallFraud; }
        public double getF1ScoreFraud() { return f1ScoreFraud; }
        public double getAuc() { return auc; }
        public double getPrecisionNormal() { return precisionNormal; }
        public double getRecallNormal() { return recallNormal; }
        public double getF1ScoreNormal() { return f1ScoreNormal; }
    }
}
