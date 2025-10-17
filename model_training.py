"""
Machine Learning Models for Credit Card Fraud Detection
Implements Logistic Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
# import xgboost as xgb  # Commented out due to OpenMP dependency
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModels:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_score = 0
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        
        # Initialize model with class balancing
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            solver='liblinear'
        )
        
        # Train model
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store model and scores
        self.models['Logistic Regression'] = lr_model
        self.model_scores['Logistic Regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Logistic Regression Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return lr_model, y_pred, y_pred_proba
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Initialize model with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store model and scores
        self.models['Random Forest'] = rf_model
        self.model_scores['Random Forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Random Forest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return rf_model, y_pred, y_pred_proba, feature_importance
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n=== TRAINING XGBOOST ===")
        
        # Initialize model with optimized parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle class imbalance
        )
        
        # Train model
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store model and scores
        self.models['XGBoost'] = xgb_model
        self.model_scores['XGBoost'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"XGBoost Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return xgb_model, y_pred, y_pred_proba, feature_importance
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance"""
        print("\n=== TRAINING ALL MODELS ===")
        
        # Train Logistic Regression
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train XGBoost (commented out due to OpenMP dependency)
        # self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Find best model
        self.find_best_model()
        
        return self.models, self.model_scores
    
    def find_best_model(self):
        """Find the best performing model based on F1-score"""
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, scores in self.model_scores.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': scores['accuracy'],
                'Precision': scores['precision'],
                'Recall': scores['recall'],
                'F1-Score': scores['f1'],
                'AUC': scores['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("Model Performance Comparison:")
        print(comparison_df.round(4))
        
        # Find best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        self.best_score = comparison_df.iloc[0]['F1-Score']
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {self.best_score:.4f})")
        
        return comparison_df
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        if not self.model_scores:
            print("No models trained yet!")
            return
        
        # Prepare data for plotting
        models = list(self.model_scores.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.model_scores[model][metric] for model in models]
            axes[i].bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for all models"""
        if not self.model_scores:
            print("No models trained yet!")
            return
        
        fig, axes = plt.subplots(1, len(self.model_scores), figsize=(5 * len(self.model_scores), 4))
        if len(self.model_scores) == 1:
            axes = [axes]
        
        for i, (model_name, scores) in enumerate(self.model_scores.items()):
            cm = confusion_matrix(y_test, scores['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Normal', 'Fraud'])
            axes[i].set_yticklabels(['Normal', 'Fraud'])
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        if not self.model_scores:
            print("No models trained yet!")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, scores in self.model_scores.items():
            fpr, tpr, _ = roc_curve(y_test, scores['probabilities'])
            auc = scores['auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, y_test):
        """Plot Precision-Recall curves for all models"""
        if not self.model_scores:
            print("No models trained yet!")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, scores in self.model_scores.items():
            precision, recall, _ = precision_recall_curve(y_test, scores['probabilities'])
            
            plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation for all models"""
        print(f"\n=== CROSS-VALIDATION ({cv_folds} FOLDS) ===")
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nCross-validating {model_name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
            
            cv_results[model_name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_models(self):
        """Save all trained models"""
        print(f"\n=== SAVING MODELS ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
            filepath = os.path.join(self.models_dir, filename)
            
            joblib.dump(model, filepath)
            print(f"Saved {model_name} to {filepath}")
        
        # Save model scores
        scores_filepath = os.path.join(self.models_dir, f"model_scores_{timestamp}.joblib")
        joblib.dump(self.model_scores, scores_filepath)
        print(f"Saved model scores to {scores_filepath}")
    
    def load_models(self, model_paths):
        """Load pre-trained models"""
        print(f"\n=== LOADING MODELS ===")
        
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                self.models[model_name] = joblib.load(path)
                print(f"Loaded {model_name} from {path}")
            else:
                print(f"Model file not found: {path}")
    
    def predict_fraud(self, X, model_name=None):
        """Predict fraud using specified model or best model"""
        if model_name is None:
            model_name = list(self.models.keys())[0] if self.models else None
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        if not self.model_scores:
            return "No models trained yet!"
        
        summary = "=== MODEL SUMMARY ===\n"
        for model_name, scores in self.model_scores.items():
            summary += f"\n{model_name}:\n"
            summary += f"  Accuracy: {scores['accuracy']:.4f}\n"
            summary += f"  Precision: {scores['precision']:.4f}\n"
            summary += f"  Recall: {scores['recall']:.4f}\n"
            summary += f"  F1-Score: {scores['f1']:.4f}\n"
            summary += f"  AUC: {scores['auc']:.4f}\n"
        
        return summary

# Example usage
if __name__ == "__main__":
    # This would be used with the preprocessed data
    print("Model training module loaded successfully!")
    print("Use with preprocessed data from data_preprocessing.py")
