"""
FastAPI application for credit risk prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
import os
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Credit Risk Model API",
    description="Predict credit risk probability for BNPL customers",
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

# Global variables for model and feature columns
model = None
feature_columns = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model, feature_columns
    
    model_path = "models/best_model.pkl"
    features_path = "models/feature_columns.json"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f" Model loaded from {model_path}")
    else:
        print(f" Warning: Model not found at {model_path}")
    
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feature_columns = json.load(f)
        print(f" Feature columns loaded: {len(feature_columns)} features")
    else:
        print(f" Warning: Feature columns not found at {features_path}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Model API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": feature_columns is not None
    }


def calculate_features(transactions):
    """Calculate features from transaction data"""
    df = pd.DataFrame([t.dict() for t in transactions])
    
    # Basic aggregations
    features = {
        'total_amount': df['amount'].sum(),
        'avg_amount': df['amount'].mean(),
        'std_amount': df['amount'].std() if len(df) > 1 else 0,
        'min_amount': df['amount'].min(),
        'max_amount': df['amount'].max(),
        'transaction_count': len(df),
        'fraud_count': df['fraud_result'].sum(),
        'fraud_rate': df['fraud_result'].mean(),
        'unique_product_categories': df['product_category'].nunique() if 'product_category' in df else 0,
        'unique_channels': df['channel_id'].nunique() if 'channel_id' in df else 0,
        'avg_amount_per_transaction': df['amount'].mean(),
        'monetary_variability': df['amount'].std() / (df['amount'].mean() + 1e-6) if len(df) > 1 else 0,
        'log_total_amount': np.log1p(df['amount'].sum()),
        'log_avg_amount': np.log1p(df['amount'].mean()),
        'avg_transaction_hour': pd.to_datetime(df['transaction_start_time']).dt.hour.mean(),
        'customer_lifespan_days': (pd.to_datetime(df['transaction_start_time']).max() - 
                                   pd.to_datetime(df['transaction_start_time']).min()).days,
        'night_transaction_ratio': ((pd.to_datetime(df['transaction_start_time']).dt.hour >= 22) | 
                                    (pd.to_datetime(df['transaction_start_time']).dt.hour <= 5)).mean(),
        'weekend_transaction_ratio': (pd.to_datetime(df['transaction_start_time']).dt.dayofweek >= 5).mean()
    }
    
    return features


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict credit risk probability for a customer"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Calculate features
        features = calculate_features(request.transactions)
        
        # Create DataFrame with correct feature order
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        if feature_columns:
            for col in feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[feature_columns]
        
        # Get prediction
        risk_probability = float(model.predict_proba(feature_df)[0, 1])
        
        # Calculate credit score (300-850 range, higher = better)
        credit_score = int(850 - (risk_probability * 550))
        
        # Determine risk category and loan recommendation
        if risk_probability < 0.3:
            risk_category = "Low Risk"
            recommended_amount = 10000
            recommended_duration = 12
        elif risk_probability < 0.6:
            risk_category = "Medium Risk"
            recommended_amount = 5000
            recommended_duration = 9
        else:
            risk_category = "High Risk"
            recommended_amount = 2000
            recommended_duration = 6
        
        return PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=risk_probability,
            credit_score=credit_score,
            risk_category=risk_category,
            recommended_loan_amount=recommended_amount,
            recommended_loan_duration_months=recommended_duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
