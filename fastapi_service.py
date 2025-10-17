"""
FastAPI Service for Credit Card Fraud Detection
Provides REST API endpoints for fraud prediction and model management
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
try:
    from model_training import FraudDetectionModels
    from data_preprocessing import DataPreprocessor
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure to run the full pipeline first: python main.py --full")
    FraudDetectionModels = None
    DataPreprocessor = None
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessor
fraud_models = None
preprocessor = None
model_loaded = False

# Pydantic models for request/response
class TransactionFeatures(BaseModel):
    """Transaction features for fraud prediction"""
    Time: float = Field(..., description="Time in seconds since first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount")

class PredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    model_used: str = Field(..., description="Model used for prediction")
    confidence: float = Field(..., description="Confidence in prediction")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    transactions: List[TransactionFeatures] = Field(..., description="List of transactions to predict")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    fraud_count: int = Field(..., description="Number of transactions predicted as fraud")

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    is_loaded: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    available_models: List[str]
    api_version: str

# Dependency to check if models are loaded
def get_models():
    global fraud_models, model_loaded
    if not model_loaded or fraud_models is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")
    return fraud_models

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model_loaded, fraud_models
    
    available_models = []
    if fraud_models and fraud_models.models:
        available_models = list(fraud_models.models.keys())
    
    return HealthResponse(
        status="healthy" if model_loaded else "models_not_loaded",
        timestamp=datetime.now().isoformat(),
        models_loaded=model_loaded,
        available_models=available_models,
        api_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures, models: FraudDetectionModels = Depends(get_models)):
    """Predict fraud for a single transaction"""
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Get prediction from best model
        predictions, probabilities = models.predict_fraud(df)
        
        fraud_probability = probabilities[0]
        is_fraud = predictions[0] == 1
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(fraud_probability - 0.5) * 2
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=float(fraud_probability),
            risk_level=risk_level,
            model_used="Best Model",
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchPredictionRequest, models: FraudDetectionModels = Depends(get_models)):
    """Predict fraud for multiple transactions"""
    try:
        # Convert transactions to DataFrame
        transactions_data = [t.dict() for t in request.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Get predictions
        predictions, probabilities = models.predict_fraud(df)
        
        # Create response for each transaction
        prediction_responses = []
        fraud_count = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            is_fraud = pred == 1
            if is_fraud:
                fraud_count += 1
            
            # Determine risk level
            if prob < 0.3:
                risk_level = "LOW"
            elif prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Calculate confidence
            confidence = abs(prob - 0.5) * 2
            
            prediction_responses.append(PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=float(prob),
                risk_level=risk_level,
                model_used="Best Model",
                confidence=float(confidence)
            ))
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_transactions=len(request.transactions),
            fraud_count=fraud_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def get_models_info(models: FraudDetectionModels = Depends(get_models)):
    """Get information about available models"""
    try:
        model_info = []
        
        for model_name, scores in models.model_scores.items():
            model_info.append(ModelInfo(
                model_name=model_name,
                accuracy=float(scores['accuracy']),
                precision=float(scores['precision']),
                recall=float(scores['recall']),
                f1_score=float(scores['f1']),
                auc=float(scores['auc']),
                is_loaded=model_name in models.models
            ))
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/models/train")
async def train_models(data_file: str = "data/creditcard.csv"):
    """Train all models with the dataset"""
    global fraud_models, model_loaded
    
    try:
        logger.info("Starting model training...")
        
        # Initialize preprocessor and models
        preprocessor = DataPreprocessor()
        fraud_models = FraudDetectionModels()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        processed_data = preprocessor.preprocess_pipeline(data_file)
        
        if not processed_data:
            raise HTTPException(status_code=400, detail="Failed to preprocess data")
        
        # Train models
        logger.info("Training models...")
        fraud_models.train_all_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        
        # Save models
        fraud_models.save_models()
        
        model_loaded = True
        
        return {
            "message": "Models trained successfully",
            "models": list(fraud_models.models.keys()),
            "best_model": fraud_models.best_model.__class__.__name__ if fraud_models.best_model else None,
            "best_score": float(fraud_models.best_score) if fraud_models.best_score else None
        }
        
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/models/load")
async def load_models(models_dir: str = "models"):
    """Load pre-trained models"""
    global fraud_models, model_loaded
    
    try:
        fraud_models = FraudDetectionModels(models_dir)
        
        # Look for model files
        model_files = {}
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.joblib') and not file.startswith('model_scores'):
                    model_name = file.split('_')[0].replace('_', ' ').title()
                    model_files[model_name] = os.path.join(models_dir, file)
        
        if not model_files:
            raise HTTPException(status_code=404, detail="No model files found")
        
        # Load models
        fraud_models.load_models(model_files)
        
        # Load scores if available
        scores_files = [f for f in os.listdir(models_dir) if f.startswith('model_scores')]
        if scores_files:
            latest_scores = max(scores_files)
            fraud_models.model_scores = joblib.load(os.path.join(models_dir, latest_scores))
        
        model_loaded = True
        
        return {
            "message": "Models loaded successfully",
            "models": list(fraud_models.models.keys()),
            "loaded_from": models_dir
        }
        
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.get("/models/performance")
async def get_model_performance(models: FraudDetectionModels = Depends(get_models)):
    """Get detailed model performance metrics"""
    try:
        if not models.model_scores:
            raise HTTPException(status_code=404, detail="No model performance data available")
        
        return {
            "model_performance": models.model_scores,
            "best_model": models.best_model.__class__.__name__ if models.best_model else None,
            "best_score": float(models.best_score) if models.best_score else None
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@app.get("/stats")
async def get_fraud_stats(models: FraudDetectionModels = Depends(get_models)):
    """Get fraud detection statistics"""
    try:
        if not models.model_scores:
            raise HTTPException(status_code=404, detail="No model data available")
        
        # Calculate average metrics across all models
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = [scores[metric] for scores in models.model_scores.values()]
            avg_metrics[f'avg_{metric}'] = float(np.mean(values))
            avg_metrics[f'best_{metric}'] = float(max(values))
        
        return {
            "average_metrics": avg_metrics,
            "total_models": len(models.model_scores),
            "available_models": list(models.model_scores.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting fraud stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get fraud stats: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global fraud_models, model_loaded
    
    logger.info("Starting Credit Card Fraud Detection API...")
    
    # Try to load existing models
    try:
        fraud_models = FraudDetectionModels()
        model_files = {}
        
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith('.joblib') and not file.startswith('model_scores'):
                    model_name = file.split('_')[0].replace('_', ' ').title()
                    model_files[model_name] = os.path.join("models", file)
        
        if model_files:
            fraud_models.load_models(model_files)
            model_loaded = True
            logger.info(f"Loaded {len(model_files)} models on startup")
        else:
            logger.info("No pre-trained models found. Use /models/train to train models.")
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        model_loaded = False

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
