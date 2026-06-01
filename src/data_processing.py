"""
Feature Engineering Module for Credit Risk Model
Task 4: Proxy Target Variable Engineering with RFM + K-Means Clustering
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate required columns exist in dataset"""
    
    REQUIRED_COLUMNS = ['CustomerId', 'Amount', 'TransactionStartTime', 
                        'TransactionId', 'FraudResult']
    
    @staticmethod
    def validate(df):
        missing_cols = [col for col in DataValidator.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("Data validation passed")
        return True


class RFMFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts Recency, Frequency, Monetary features from transaction data"""
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        self.customer_rfm = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        if self.snapshot_date:
            snapshot = pd.to_datetime(self.snapshot_date)
        else:
            snapshot = df['TransactionStartTime'].max()
            
        self.customer_rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot - x.max()).days,
            'TransactionId': 'count',
            'Amount': lambda x: x.abs().sum()
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        })
        
        logger.info(f"RFM features extracted for {len(self.customer_rfm)} customers")
        return self.customer_rfm.reset_index()


class RFMScaler(BaseEstimator, TransformerMixin):
    """Scales RFM features for clustering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X, y=None):
        rfm_features = X[['Recency', 'Frequency', 'Monetary']].values
        self.scaler.fit(rfm_features)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("RFMScaler must be fitted before transform. Call fit() first.")
        X_scaled = X.copy()
        rfm_features = X_scaled[['Recency', 'Frequency', 'Monetary']].values
        scaled = self.scaler.transform(rfm_features)
        X_scaled['Recency_scaled'] = scaled[:, 0]
        X_scaled['Frequency_scaled'] = scaled[:, 1]
        X_scaled['Monetary_scaled'] = scaled[:, 2]
        logger.info("RFM features scaled")
        return X_scaled


class HighRiskLabeler(BaseEstimator, TransformerMixin):
    """Clusters customers based on RFM and assigns high-risk label"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_risk_mapping = None
        self.cluster_characteristics = None
        self.fitted = False
        
    def fit(self, X, y=None):
        rfm_scaled = X[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']].values
        self.kmeans.fit(rfm_scaled)
        
        cluster_centers = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
        )
        
        cluster_centers['risk_score'] = (
            cluster_centers['Recency_scaled'] - 
            cluster_centers['Frequency_scaled'] - 
            cluster_centers['Monetary_scaled']
        )
        
        high_risk_cluster = cluster_centers['risk_score'].idxmax()
        self.cluster_risk_mapping = {i: 1 if i == high_risk_cluster else 0 for i in range(self.n_clusters)}
        
        self.cluster_characteristics = cluster_centers
        self.fitted = True
        logger.info(f"High-risk cluster identified: Cluster {high_risk_cluster}")
        logger.info(f"Cluster risk scores: {cluster_centers['risk_score'].to_dict()}")
        
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("HighRiskLabeler must be fitted before transform. Call fit() first.")
        X_result = X.copy()
        rfm_scaled = X_result[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']].values
        clusters = self.kmeans.predict(rfm_scaled)
        X_result['Cluster'] = clusters
        X_result['is_high_risk'] = X_result['Cluster'].map(self.cluster_risk_mapping)
        
        high_risk_count = X_result['is_high_risk'].sum()
        logger.info(f"High-risk customers: {high_risk_count} ({high_risk_count/len(X_result)*100:.2f}%)")
        
        return X_result


class AggregateFeatureBuilder(BaseEstimator, TransformerMixin):
    """Builds aggregate features at customer level with error handling"""
    
    def __init__(self, cap_outliers=True, outlier_percentile=99):
        self.cap_outliers = cap_outliers
        self.outlier_percentile = outlier_percentile
        self.upper_cap = None
        
    def fit(self, X, y=None):
        if self.cap_outliers and 'Amount' in X.columns:
            self.upper_cap = X['Amount'].abs().quantile(self.outlier_percentile / 100)
            logger.info(f"Outlier cap set at {self.outlier_percentile}th percentile: {self.upper_cap:.2f}")
        return self
    
    def transform(self, X):
        df = X.copy()
        df['Amount_abs'] = df['Amount'].abs()
        
        if self.cap_outliers and self.upper_cap:
            df['Amount_abs'] = df['Amount_abs'].clip(upper=self.upper_cap)
            logger.info(f"Capped outlier values")
        
        customer_features = df.groupby('CustomerId').agg({
            'Amount_abs': ['sum', 'mean', 'std', 'min', 'max'],
            'TransactionId': 'count',
            'FraudResult': ['sum', 'mean'],
            'ProductCategory': lambda x: x.nunique(),
            'ChannelId': lambda x: x.nunique()
        }).round(2)
        
        customer_features.columns = [
            'total_amount', 'avg_amount', 'std_amount', 'min_amount', 'max_amount',
            'transaction_count', 'fraud_count', 'fraud_rate',
            'unique_product_categories', 'unique_channels'
        ]
        
        customer_features['std_amount'] = customer_features['std_amount'].fillna(0)
        
        customer_features['avg_amount_per_transaction'] = (
            customer_features['total_amount'] / customer_features['transaction_count']
        )
        customer_features['monetary_variability'] = (
            customer_features['std_amount'] / (customer_features['avg_amount'] + 1e-6)
        )
        
        for col in ['total_amount', 'avg_amount']:
            customer_features[f'log_{col}'] = np.log1p(customer_features[col])
        
        logger.info(f"Created {len(customer_features.columns)} aggregate features")
        
        return customer_features.reset_index()


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features from transaction timestamps"""
    
    def __init__(self, night_hours=(22, 5)):
        self.night_hours = night_hours
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        time_features = df.groupby('CustomerId').agg({
            'TransactionStartTime': [
                lambda x: x.dt.hour.mean(),
                lambda x: (x.max() - x.min()).days
            ]
        })
        
        time_features.columns = ['avg_transaction_hour', 'customer_lifespan_days']
        
        df['hour'] = df['TransactionStartTime'].dt.hour
        night_start, night_end = self.night_hours
        if night_start > night_end:
            df['is_night'] = ((df['hour'] >= night_start) | (df['hour'] <= night_end)).astype(int)
        else:
            df['is_night'] = ((df['hour'] >= night_start) & (df['hour'] <= night_end)).astype(int)
        
        night_txn = df.groupby('CustomerId')['is_night'].mean().rename('night_transaction_ratio')
        time_features = time_features.join(night_txn)
        
        df['is_weekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int)
        weekend_txn = df.groupby('CustomerId')['is_weekend'].mean().rename('weekend_transaction_ratio')
        time_features = time_features.join(weekend_txn)
        
        time_features['customer_lifespan_days'] = time_features['customer_lifespan_days'].clip(lower=0)
        time_features = time_features.fillna(0)
        
        logger.info(f"Created {len(time_features.columns)} time-based features")
        
        return time_features.reset_index()


class FeatureProcessor:
    """Main feature engineering pipeline with RFM-based target variable"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.data_path = self.config.get('data_path', 'data/raw/')
        self.output_path = self.config.get('output_path', 'data/processed/processed_data.csv')
        
        self.agg_builder = AggregateFeatureBuilder(
            cap_outliers=self.config.get('cap_outliers', True),
            outlier_percentile=self.config.get('outlier_percentile', 99)
        )
        self.time_extractor = TimeFeatureExtractor(
            night_hours=self.config.get('night_hours', (22, 5))
        )
        self.rfm_extractor = RFMFeatureExtractor()
        self.rfm_scaler = RFMScaler()
        self.high_risk_labeler = HighRiskLabeler(
            n_clusters=self.config.get('n_clusters', 3),
            random_state=self.config.get('random_state', 42)
        )
        
    def load_data(self):
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Directory {self.data_path} does not exist")
            
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            
            df = pd.read_csv(os.path.join(self.data_path, csv_files[0]))
            logger.info(f"Loaded {csv_files[0]} with shape {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def save_processed_data(self, df):
        try:
            Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Processed data saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise
    
    def process(self, df=None):
        if df is None:
            df = self.load_data()
        
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        DataValidator.validate(df)
        
        # Step 1: Build aggregate features
        logger.info("\n[Step 1] Building aggregate features...")
        agg_features = self.agg_builder.fit_transform(df)
        
        # Step 2: Extract time features
        logger.info("\n[Step 2] Extracting time features...")
        time_features = self.time_extractor.transform(df)
        
        # Step 3: Create RFM-based target variable
        logger.info("\n[Step 3] Creating RFM-based proxy target variable...")
        rfm_data = self.rfm_extractor.transform(df)
        rfm_scaled = self.rfm_scaler.fit_transform(rfm_data)  # FIXED: Use fit_transform instead of transform
        rfm_labeled = self.high_risk_labeler.fit_transform(rfm_scaled)  # FIXED: Use fit_transform
        
        # Step 4: Merge all features with target
        logger.info("\n[Step 4] Merging features with target variable...")
        final_df = agg_features.merge(time_features, on='CustomerId', how='left')
        final_df = final_df.merge(rfm_labeled[['CustomerId', 'is_high_risk', 'Recency', 'Frequency', 'Monetary', 'Cluster']], 
                                  on='CustomerId', how='left')
        
        # Step 5: Clean up
        final_df = final_df.fillna(0)
        final_df = final_df.replace([np.inf, -np.inf], 0)
        
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info(f"High-risk customers: {final_df['is_high_risk'].sum()} ({final_df['is_high_risk'].mean()*100:.2f}%)")
        logger.info(f"Features created: {list(final_df.columns)}")
        
        return final_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline with RFM Target')
    parser.add_argument('--input', type=str, default='data/raw/',
                        help='Path to raw data directory')
    parser.add_argument('--output', type=str, default='data/processed/processed_data.csv',
                        help='Path to save processed data')
    parser.add_argument('--no-cap', action='store_true',
                        help='Disable outlier capping')
    parser.add_argument('--clusters', type=int, default=3,
                        help='Number of clusters for K-Means')
    
    args = parser.parse_args()
    
    config = {
        'data_path': args.input,
        'output_path': args.output,
        'cap_outliers': not args.no_cap,
        'outlier_percentile': 99,
        'night_hours': (22, 5),
        'n_clusters': args.clusters,
        'random_state': 42
    }
    
    processor = FeatureProcessor(config)
    processed_data = processor.process()
    processor.save_processed_data(processed_data)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Output saved to: {args.output}")
    print(f"Shape: {processed_data.shape}")
    print(f"\nTarget distribution:")
    print(processed_data['is_high_risk'].value_counts())
    print(f"\nFeatures: {processed_data.columns.tolist()}")


if __name__ == "__main__":
    main()
