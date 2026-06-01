"""
Unit tests for data processing module
"""

import pandas as pd


class TestAggregateFeatureBuilder:
    """Tests for AggregateFeatureBuilder class"""

    def test_aggregate_features_returns_expected_columns(self):
        from src.data_processing import AggregateFeatureBuilder

        # Create sample data
        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 50, 150, 300],
            'TransactionId': [101, 102, 103, 104, 105],
            'FraudResult': [0, 0, 1, 0, 0],
            'ProductCategory': ['A', 'B', 'A', 'C', 'A'],
            'ChannelId': ['web', 'app', 'web', 'app', 'web']
        })

        builder = AggregateFeatureBuilder()
        builder.fit(df)
        result = builder.transform(df)

        # Check expected columns
        expected_cols = [
            'CustomerId', 'total_amount', 'avg_amount', 'std_amount',
            'min_amount', 'max_amount', 'transaction_count', 'fraud_count',
            'fraud_rate', 'unique_product_categories', 'unique_channels',
            'avg_amount_per_transaction', 'monetary_variability',
            'log_total_amount', 'log_avg_amount'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Column {col} not found"

        # Check customer 1 has correct total
        cust1 = result[result['CustomerId'] == 1]
        assert cust1['total_amount'].iloc[0] == 300  # 100 + 200

    def test_handles_single_transaction_customers(self):
        from src.data_processing import AggregateFeatureBuilder

        df = pd.DataFrame({
            'CustomerId': [1, 2],
            'Amount': [100, 200],
            'TransactionId': [101, 102],
            'FraudResult': [0, 0],
            'ProductCategory': ['A', 'B'],
            'ChannelId': ['web', 'app']
        })

        builder = AggregateFeatureBuilder()
        builder.fit(df)
        result = builder.transform(df)

        # std_amount should be 0 for single transactions (not NaN)
        assert result['std_amount'].iloc[0] == 0
        assert result['std_amount'].iloc[1] == 0


class TestRFMFeatureExtractor:
    """Tests for RFMFeatureExtractor class"""

    def test_rfm_extractor_returns_expected_columns(self):
        from src.data_processing import RFMFeatureExtractor

        df = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5),
            'TransactionId': [101, 102, 103, 104, 105],
            'Amount': [100, 200, 50, 150, 300]
        })

        extractor = RFMFeatureExtractor()
        result = extractor.transform(df)

        assert 'CustomerId' in result.columns
        assert 'Recency' in result.columns
        assert 'Frequency' in result.columns
        assert 'Monetary' in result.columns
        assert len(result) == df['CustomerId'].nunique()


def test_sample():
    """Simple test to verify pytest works"""
    assert 1 + 1 == 2
