"""
Feature Engineering Module for Credit Risk Model
Task 3: Feature engineering pipeline
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregateFeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds aggregate features at customer level"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        # Use absolute amount for monetary values
        df['Amount_abs'] = df['Amount'].abs()
        
        # Customer-level aggregations
        customer_features = df.groupby('CustomerId').agg({
            'Amount_abs': ['sum', 'mean', 'std', 'min', 'max'],
            'TransactionId': 'count',
            'FraudResult': ['sum', 'mean'],
            'ProductCategory': lambda x: x.nunique(),
            'ChannelId': lambda x: x.nunique()
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'total_amount', 'avg_amount', 'std_amount', 'min_amount', 'max_amount',
            'transaction_count', 'fraud_count', 'fraud_rate',
            'unique_product_categories', 'unique_channels'
        ]
        
        # Handle missing std (for customers with single transaction)
        customer_features['std_amount'] = customer_features['std_amount'].fillna(0)
        
        # Derived features
        customer_features['avg_amount_per_transaction'] = (
            customer_features['total_amount'] / customer_features['transaction_count']
        )
        customer_features['monetary_variability'] = (
            customer_features['std_amount'] / (customer_features['avg_amount'] + 1e-6)
        )
        
        return customer_features.reset_index()


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features from transaction timestamps"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Time-based aggregations
        time_features = df.groupby('CustomerId').agg({
            'TransactionStartTime': [
                lambda x: x.dt.hour.mean(),  # Average hour of transactions
                lambda x: (x.max() - x.min()).days  # Customer lifespan in days
            ]
        })
        
        time_features.columns = ['avg_transaction_hour', 'customer_lifespan_days']
        
        # Night transactions ratio (10 PM to 5 AM)
        df['hour'] = df['TransactionStartTime'].dt.hour
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        night_txn = df.groupby('CustomerId')['is_night'].mean().rename('night_transaction_ratio')
        
        time_features = time_features.join(night_txn)
        
        # Weekend transaction ratio
        df['is_weekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int)
        weekend_txn = df.groupby('CustomerId')['is_weekend'].mean().rename('weekend_transaction_ratio')
        time_features = time_features.join(weekend_txn)
        
        # Clean up
        time_features['customer_lifespan_days'] = time_features['customer_lifespan_days'].clip(lower=0)
        time_features = time_features.fillna(0)
        
        return time_features.reset_index()


class FeatureProcessor:
    """Main feature engineering pipeline"""
    
    def __init__(self):
        self.agg_builder = AggregateFeatureBuilder()
        self.time_extractor = TimeFeatureExtractor()
    
    def process(self, df):
        """Process data to create features"""
        logger.info("Starting feature engineering...")
        
        # Build aggregate features
        logger.info("Building aggregate features...")
        agg_features = self.agg_builder.transform(df)
        
        # Extract time features
        logger.info("Extracting time features...")
        time_features = self.time_extractor.transform(df)
        
        # Merge all features
        final_df = agg_features.merge(time_features, on='CustomerId', how='left')
        
        # Handle any remaining missing values
        final_df = final_df.fillna(0)
        final_df = final_df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature engineering complete. Shape: {final_df.shape}")
        logger.info(f"Features created: {list(final_df.columns)}")
        
        return final_df


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/raw/xe.csv')
    
    # Process features
    processor = FeatureProcessor()
    features = processor.process(df)
    
    # Save features
    features.to_csv('data/processed/features.csv', index=False)
    print(f"Features saved to data/processed/features.csv")
    print(f"Final feature set shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
