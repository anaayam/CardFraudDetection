package com.frauddetection.model;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * Trains multiple machine learning models for fraud detection
 */
public class ModelTrainer {
    
    /**
     * Train multiple models on the dataset
     * @param dataset Training dataset
     * @return List of trained classifiers
     */
    public List<Classifier> trainModels(Instances dataset) {
        List<Classifier> models = new ArrayList<>();
        
        System.out.println("Training multiple models...");
        
        try {
            // 1. Random Forest
            System.out.println("Training Random Forest...");
            RandomForest randomForest = new RandomForest();
            randomForest.setNumIterations(100);
            randomForest.setNumFeatures(0); // Use all features
            randomForest.buildClassifier(dataset);
            models.add(randomForest);
            
            // 2. J48 Decision Tree
            System.out.println("Training J48 Decision Tree...");
            J48 j48 = new J48();
            j48.setConfidenceFactor(0.25f);
            j48.setMinNumObj(2);
            j48.buildClassifier(dataset);
            models.add(j48);
            
            // 3. Naive Bayes
            System.out.println("Training Naive Bayes...");
            NaiveBayes naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(dataset);
            models.add(naiveBayes);
            
            // 4. Support Vector Machine (SMO)
            System.out.println("Training Support Vector Machine...");
            SMO smo = new SMO();
            smo.setC(1.0);
            smo.buildClassifier(dataset);
            models.add(smo);
            
            // 5. Logistic Regression
            System.out.println("Training Logistic Regression...");
            Logistic logistic = new Logistic();
            logistic.setMaxIts(-1); // No limit on iterations
            logistic.buildClassifier(dataset);
            models.add(logistic);
            
            System.out.println("All models trained successfully!");
            
        } catch (Exception e) {
            System.err.println("Error training models: " + e.getMessage());
            e.printStackTrace();
        }
        
        return models;
    }
    
    /**
     * Train a single model with custom parameters
     * @param dataset Training dataset
     * @param modelType Type of model to train
     * @return Trained classifier
     */
    public Classifier trainSingleModel(Instances dataset, String modelType) {
        try {
            switch (modelType.toLowerCase()) {
                case "randomforest":
                    RandomForest rf = new RandomForest();
                    rf.setNumIterations(100);
                    rf.setNumFeatures(0);
                    rf.buildClassifier(dataset);
                    return rf;
                    
                case "j48":
                    J48 j48 = new J48();
                    j48.setConfidenceFactor(0.25f);
                    j48.setMinNumObj(2);
                    j48.buildClassifier(dataset);
                    return j48;
                    
                case "naivebayes":
                    NaiveBayes nb = new NaiveBayes();
                    nb.buildClassifier(dataset);
                    return nb;
                    
                case "smo":
                    SMO smo = new SMO();
                    smo.setC(1.0);
                    smo.buildClassifier(dataset);
                    return smo;
                    
                case "logistic":
                    Logistic log = new Logistic();
                    log.setMaxIts(-1);
                    log.buildClassifier(dataset);
                    return log;
                    
                default:
                    throw new IllegalArgumentException("Unknown model type: " + modelType);
            }
        } catch (Exception e) {
            System.err.println("Error training " + modelType + ": " + e.getMessage());
            throw new RuntimeException(e);
        }
    }
    
    /**
     * Get model names for the trained models
     * @return List of model names
     */
    public List<String> getModelNames() {
        List<String> names = new ArrayList<>();
        names.add("Random Forest");
        names.add("J48 Decision Tree");
        names.add("Naive Bayes");
        names.add("Support Vector Machine");
        names.add("Logistic Regression");
        return names;
    }
}
