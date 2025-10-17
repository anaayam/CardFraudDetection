package com.frauddetection.entity;

/**
 * Represents a credit card transaction with all relevant features
 */
public class Transaction {
    private double time;
    private double amount;
    private double[] vFeatures; // V1-V28 PCA features
    private int classLabel; // 0 for normal, 1 for fraud
    
    public Transaction() {
        this.vFeatures = new double[28];
    }
    
    public Transaction(double time, double amount, double[] vFeatures, int classLabel) {
        this.time = time;
        this.amount = amount;
        this.vFeatures = vFeatures.clone();
        this.classLabel = classLabel;
    }
    
    // Getters and Setters
    public double getTime() {
        return time;
    }
    
    public void setTime(double time) {
        this.time = time;
    }
    
    public double getAmount() {
        return amount;
    }
    
    public void setAmount(double amount) {
        this.amount = amount;
    }
    
    public double[] getVFeatures() {
        return vFeatures.clone();
    }
    
    public void setVFeatures(double[] vFeatures) {
        this.vFeatures = vFeatures.clone();
    }
    
    public int getClassLabel() {
        return classLabel;
    }
    
    public void setClassLabel(int classLabel) {
        this.classLabel = classLabel;
    }
    
    /**
     * Get all features as a single array for machine learning
     * @return array containing time, amount, and V1-V28 features
     */
    public double[] getAllFeatures() {
        double[] allFeatures = new double[30]; // time + amount + 28 V features
        allFeatures[0] = time;
        allFeatures[1] = amount;
        System.arraycopy(vFeatures, 0, allFeatures, 2, 28);
        return allFeatures;
    }
    
    /**
     * Get features without time and amount (only V1-V28)
     * @return array containing only V1-V28 features
     */
    public double[] getVFeaturesOnly() {
        return vFeatures.clone();
    }
    
    public boolean isFraud() {
        return classLabel == 1;
    }
    
    @Override
    public String toString() {
        return String.format("Transaction{time=%.2f, amount=%.2f, fraud=%s}", 
                           time, amount, isFraud());
    }
}
