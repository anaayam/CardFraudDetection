package com.frauddetection;

import com.frauddetection.entity.Transaction;
import com.frauddetection.model.DataProcessor;
import com.frauddetection.model.ModelTrainer;
import com.frauddetection.model.FraudPredictor;
import com.frauddetection.util.DataUtils;
import weka.core.Instances;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Unit tests for the fraud detection system
 */
public class FraudDetectionTest {
    
    private DataProcessor processor;
    private ModelTrainer trainer;
    
    @BeforeEach
    void setUp() {
        processor = new DataProcessor();
        trainer = new ModelTrainer();
    }
    
    @Test
    void testTransactionCreation() {
        double[] vFeatures = new double[28];
        for (int i = 0; i < 28; i++) {
            vFeatures[i] = i * 0.1;
        }
        
        Transaction transaction = new Transaction(0.0, 100.0, vFeatures, 0);
        
        assertEquals(0.0, transaction.getTime());
        assertEquals(100.0, transaction.getAmount());
        assertEquals(0, transaction.getClassLabel());
        assertFalse(transaction.isFraud());
        
        double[] allFeatures = transaction.getAllFeatures();
        assertEquals(30, allFeatures.length);
        assertEquals(0.0, allFeatures[0]); // Time
        assertEquals(100.0, allFeatures[1]); // Amount
    }
    
    @Test
    void testDataUtils() {
        List<double[]> data = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        // Create sample data
        for (int i = 0; i < 100; i++) {
            double[] features = {i * 0.1, i * 0.2, i * 0.3};
            data.add(features);
            labels.add(i % 2); // Alternating labels
        }
        
        // Test normalization
        List<double[]> normalized = DataUtils.normalizeFeatures(data);
        assertEquals(data.size(), normalized.size());
        assertEquals(data.get(0).length, normalized.get(0).length);
        
        // Test train-test split
        List<double[]>[] split = DataUtils.trainTestSplit(data, labels, 0.2);
        assertEquals(4, split.length);
        assertEquals(80, split[0].size()); // Train data
        assertEquals(20, split[2].size()); // Test data
        
        // Test class distribution
        int[] distribution = DataUtils.getClassDistribution(labels);
        assertEquals(2, distribution.length);
        assertEquals(50, distribution[0]);
        assertEquals(50, distribution[1]);
    }
    
    @Test
    void testModelTrainer() {
        // Create a simple dataset for testing
        Instances dataset = createTestDataset();
        
        // Test training multiple models
        List<weka.classifiers.Classifier> models = trainer.trainModels(dataset);
        assertNotNull(models);
        assertTrue(models.size() > 0);
        
        // Test training single model
        weka.classifiers.Classifier singleModel = trainer.trainSingleModel(dataset, "naivebayes");
        assertNotNull(singleModel);
        
        // Test model names
        List<String> names = trainer.getModelNames();
        assertNotNull(names);
        assertTrue(names.size() > 0);
    }
    
    @Test
    void testFraudPredictor() {
        // Create test dataset and models
        Instances dataset = createTestDataset();
        List<weka.classifiers.Classifier> models = trainer.trainModels(dataset);
        List<String> modelNames = trainer.getModelNames();
        
        FraudPredictor predictor = new FraudPredictor(models, modelNames, dataset);
        
        // Test prediction with sample features
        double[] features = new double[30];
        for (int i = 0; i < 30; i++) {
            features[i] = i * 0.1;
        }
        
        FraudPredictor.PredictionResult result = predictor.predictFraud(features);
        assertNotNull(result);
        assertTrue(result.getFraudProbability() >= 0.0 && result.getFraudProbability() <= 1.0);
        
        // Test V-features only prediction
        double[] vFeatures = new double[28];
        for (int i = 0; i < 28; i++) {
            vFeatures[i] = i * 0.1;
        }
        
        FraudPredictor.PredictionResult vResult = predictor.predictFraudVFeatures(vFeatures);
        assertNotNull(vResult);
    }
    
    @Test
    void testPredictionResult() {
        List<Double> probabilities = List.of(0.3, 0.7, 0.5);
        List<Boolean> predictions = List.of(false, true, false);
        List<String> names = List.of("Model1", "Model2", "Model3");
        
        FraudPredictor.PredictionResult result = new FraudPredictor.PredictionResult(
            true, 0.5, probabilities, predictions, names
        );
        
        assertTrue(result.isFraud());
        assertEquals(0.5, result.getFraudProbability());
        assertEquals("MEDIUM", result.getRiskLevel());
        assertNotNull(result.toString());
    }
    
    /**
     * Create a simple test dataset for unit testing
     */
    private Instances createTestDataset() {
        try {
            // Create attributes
            weka.core.FastVector attributes = new weka.core.FastVector();
            for (int i = 0; i < 30; i++) {
                attributes.addElement(new weka.core.Attribute("feature_" + i));
            }
            
            weka.core.FastVector classValues = new weka.core.FastVector();
            classValues.addElement("0");
            classValues.addElement("1");
            attributes.addElement(new weka.core.Attribute("class", classValues));
            
            // Create dataset
            Instances dataset = new Instances("TestDataset", attributes, 0);
            dataset.setClassIndex(30);
            
            // Add sample instances
            for (int i = 0; i < 100; i++) {
                weka.core.Instance instance = new weka.core.DenseInstance(31);
                instance.setDataset(dataset);
                
                for (int j = 0; j < 30; j++) {
                    instance.setValue(j, Math.random());
                }
                instance.setValue(30, String.valueOf(i % 2));
                
                dataset.add(instance);
            }
            
            return dataset;
        } catch (Exception e) {
            throw new RuntimeException("Error creating test dataset", e);
        }
    }
}
