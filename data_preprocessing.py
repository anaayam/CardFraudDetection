"""
Data Cleaning and Preprocessing Module for Credit Card Fraud Detection
Based on Kaggle Credit Card Fraud Detection dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load the credit card fraud detection dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Perform initial data exploration"""
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        print(f"\nMissing values per column:")
        print(missing_data[missing_data > 0])
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        # Basic statistics
        print(f"\nBasic statistics:")
        print(df.describe())
        
        return df
    
    def analyze_class_distribution(self, df):
        """Analyze the class distribution (fraud vs normal)"""
        print("\n=== CLASS DISTRIBUTION ANALYSIS ===")
        
        class_counts = df['Class'].value_counts()
        class_percentages = df['Class'].value_counts(normalize=True) * 100
        
        print("Class distribution:")
        for class_val in [0, 1]:
            count = class_counts[class_val]
            percentage = class_percentages[class_val]
            label = "Normal" if class_val == 0 else "Fraud"
            print(f"{label}: {count:,} ({percentage:.2f}%)")
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar', color=['green', 'red'])
        plt.title('Class Distribution (Count)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
        
        plt.subplot(1, 2, 2)
        class_percentages.plot(kind='bar', color=['green', 'red'])
        plt.title('Class Distribution (Percentage)')
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        return class_counts, class_percentages
    
    def analyze_time_distribution(self, df):
        """Analyze transaction time patterns"""
        print("\n=== TIME DISTRIBUTION ANALYSIS ===")
        
        # Convert time to hours for better visualization
        df['Hour'] = df['Time'] / 3600
        
        # Analyze by class
        normal_times = df[df['Class'] == 0]['Hour']
        fraud_times = df[df['Class'] == 1]['Hour']
        
        print(f"Normal transactions time range: {normal_times.min():.1f} - {normal_times.max():.1f} hours")
        print(f"Fraud transactions time range: {fraud_times.min():.1f} - {fraud_times.max():.1f} hours")
        
        # Visualize time distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(normal_times, bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(fraud_times, bins=50, alpha=0.7, label='Fraud', color='red')
        plt.title('Transaction Time Distribution')
        plt.xlabel('Time (hours)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(normal_times, bins=50, alpha=0.7, label='Normal', color='green', density=True)
        plt.hist(fraud_times, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        plt.title('Transaction Time Distribution (Density)')
        plt.xlabel('Time (hours)')
        plt.ylabel('Density')
        plt.legend()
        
        # Hourly analysis
        hourly_stats = df.groupby(['Hour', 'Class']).agg({
            'Amount': ['count', 'sum', 'mean', 'std']
        }).round(2)
        
        plt.subplot(2, 2, 3)
        hourly_normal = df[df['Class'] == 0].groupby('Hour')['Amount'].sum()
        hourly_fraud = df[df['Class'] == 1].groupby('Hour')['Amount'].sum()
        plt.plot(hourly_normal.index, hourly_normal.values, label='Normal', color='green')
        plt.plot(hourly_fraud.index, hourly_fraud.values, label='Fraud', color='red')
        plt.title('Total Amount by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Total Amount')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        hourly_count_normal = df[df['Class'] == 0].groupby('Hour')['Amount'].count()
        hourly_count_fraud = df[df['Class'] == 1].groupby('Hour')['Amount'].count()
        plt.plot(hourly_count_normal.index, hourly_count_normal.values, label='Normal', color='green')
        plt.plot(hourly_count_fraud.index, hourly_count_fraud.values, label='Fraud', color='red')
        plt.title('Transaction Count by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return hourly_stats
    
    def analyze_amount_distribution(self, df):
        """Analyze transaction amount patterns"""
        print("\n=== AMOUNT DISTRIBUTION ANALYSIS ===")
        
        normal_amounts = df[df['Class'] == 0]['Amount']
        fraud_amounts = df[df['Class'] == 1]['Amount']
        
        print("Normal transactions amount statistics:")
        print(normal_amounts.describe())
        print("\nFraud transactions amount statistics:")
        print(fraud_amounts.describe())
        
        # Visualize amount distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red')
        plt.title('Amount Distribution')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.boxplot([normal_amounts, fraud_amounts], labels=['Normal', 'Fraud'])
        plt.title('Amount Distribution (Box Plot)')
        plt.ylabel('Amount')
        
        plt.subplot(2, 3, 3)
        plt.hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='green', density=True)
        plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        plt.title('Amount Distribution (Density)')
        plt.xlabel('Amount')
        plt.ylabel('Density')
        plt.legend()
        
        # Log scale for better visualization
        plt.subplot(2, 3, 4)
        plt.hist(np.log1p(normal_amounts), bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(np.log1p(fraud_amounts), bins=50, alpha=0.7, label='Fraud', color='red')
        plt.title('Amount Distribution (Log Scale)')
        plt.xlabel('Log(Amount + 1)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Scatter plot: Time vs Amount
        plt.subplot(2, 3, 5)
        plt.scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'], 
                   alpha=0.5, label='Normal', color='green', s=1)
        plt.scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'], 
                   alpha=0.8, label='Fraud', color='red', s=10)
        plt.title('Time vs Amount')
        plt.xlabel('Time')
        plt.ylabel('Amount')
        plt.legend()
        
        # Amount vs Class correlation
        plt.subplot(2, 3, 6)
        correlation_data = df[['Amount', 'Class']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
        plt.title('Amount-Class Correlation')
        
        plt.tight_layout()
        plt.show()
        
        return normal_amounts.describe(), fraud_amounts.describe()
    
    def analyze_pca_features(self, df):
        """Analyze PCA-transformed features (V1-V28)"""
        print("\n=== PCA FEATURES ANALYSIS ===")
        
        # Get PCA features
        pca_features = [f'V{i}' for i in range(1, 29)]
        
        # Calculate correlations
        correlation_matrix = df[pca_features + ['Class']].corr()
        
        # Visualize correlation with Class
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        class_correlations = correlation_matrix['Class'].drop('Class').abs().sort_values(ascending=False)
        class_correlations.plot(kind='bar')
        plt.title('Feature Correlation with Class')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix (First 10 features)')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance analysis
        print("Top 10 features most correlated with Class:")
        print(class_correlations.head(10))
        
        return class_correlations
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("\n=== DATA CLEANING ===")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows")
        
        # Check for infinite values
        inf_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        print(f"Infinite values: {inf_values}")
        
        # Handle outliers in Amount (using IQR method)
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)).sum()
        print(f"Amount outliers: {outliers}")
        
        # We'll keep outliers as they might be important for fraud detection
        print("Keeping outliers as they may be important for fraud detection")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Select features (excluding Time and Class for now)
        feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        self.feature_columns = feature_columns
        
        # Create feature matrix and target
        X = df[feature_columns].copy()
        y = df['Class'].copy()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Add time-based features
        X['Time'] = df['Time']
        X['Hour'] = df['Time'] / 3600
        X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
        X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
        
        # Add amount-based features
        X['Amount_log'] = np.log1p(X['Amount'])
        X['Amount_sqrt'] = np.sqrt(X['Amount'])
        
        # Update feature columns
        self.feature_columns = X.columns.tolist()
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using RobustScaler"""
        print("\n=== FEATURE SCALING ===")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def handle_imbalance(self, X, y, method='undersample'):
        """Handle class imbalance"""
        print(f"\n=== HANDLING CLASS IMBALANCE ({method.upper()}) ===")
        
        print(f"Original class distribution:")
        print(y.value_counts())
        
        if method == 'undersample':
            # Undersample majority class
            df_combined = pd.concat([X, y], axis=1)
            
            # Separate classes
            normal = df_combined[df_combined['Class'] == 0]
            fraud = df_combined[df_combined['Class'] == 1]
            
            # Undersample normal class
            normal_undersampled = resample(normal, 
                                        replace=False, 
                                        n_samples=len(fraud), 
                                        random_state=42)
            
            # Combine undersampled data
            df_balanced = pd.concat([normal_undersampled, fraud])
            
            X_balanced = df_balanced.drop('Class', axis=1)
            y_balanced = df_balanced['Class']
            
        elif method == 'oversample':
            # Oversample minority class
            df_combined = pd.concat([X, y], axis=1)
            
            # Separate classes
            normal = df_combined[df_combined['Class'] == 0]
            fraud = df_combined[df_combined['Class'] == 1]
            
            # Oversample fraud class
            fraud_oversampled = resample(fraud, 
                                       replace=True, 
                                       n_samples=len(normal), 
                                       random_state=42)
            
            # Combine oversampled data
            df_balanced = pd.concat([normal, fraud_oversampled])
            
            X_balanced = df_balanced.drop('Class', axis=1)
            y_balanced = df_balanced['Class']
        
        print(f"Balanced class distribution:")
        print(y_balanced.value_counts())
        
        return X_balanced, y_balanced
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n=== DATA SPLITTING ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        print(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, balance_method='undersample'):
        """Complete preprocessing pipeline"""
        print("=== CREDIT CARD FRAUD DETECTION - DATA PREPROCESSING ===")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Explore data
        df = self.explore_data(df)
        
        # Analyze class distribution
        self.analyze_class_distribution(df)
        
        # Analyze time distribution
        self.analyze_time_distribution(df)
        
        # Analyze amount distribution
        self.analyze_amount_distribution(df)
        
        # Analyze PCA features
        self.analyze_pca_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Handle class imbalance
        X_balanced, y_balanced = self.handle_imbalance(X, y, method=balance_method)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_balanced, y_balanced)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n=== PREPROCESSING COMPLETE ===")
        print(f"Final training set: {X_train_scaled.shape}")
        print(f"Final test set: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline('data/creditcard.csv')
    
    if processed_data:
        print("Preprocessing completed successfully!")
        print(f"Features: {processed_data['feature_columns']}")
    else:
        print("Preprocessing failed!")
