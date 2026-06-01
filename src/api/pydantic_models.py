"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Transaction(BaseModel):
    """Transaction data for a customer"""
    amount: float = Field(..., description="Transaction amount", gt=0)
    transaction_start_time: datetime = Field(..., description="Transaction timestamp")
    product_category: Optional[str] = Field(None, description="Product category")
    channel_id: Optional[str] = Field(None, description="Channel (web, app, etc.)")
    fraud_result: int = Field(0, description="Fraud flag (1 = fraud, 0 = not fraud)", ge=0, le=1)


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    customer_id: str = Field(..., description="Unique customer identifier")
    transactions: List[Transaction] = Field(..., description="List of customer transactions", min_length=1)


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., description="Probability of default (0-1)", ge=0, le=1)
    credit_score: int = Field(..., description="Credit score (300-850)", ge=300, le=850)
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    recommended_loan_amount: float = Field(..., description="Recommended maximum loan amount")
    recommended_loan_duration_months: int = Field(..., description="Recommended loan duration in months")
