"""
Credit Risk Model - Exploratory Data Analysis
Task 2: EDA for Xente eCommerce Transaction Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("CREDIT RISK MODEL - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load data
data_path = 'data/raw/'
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')] if os.path.exists(data_path) else []

if csv_files:
    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
    print(f"Loaded: {csv_files[0]}")
    print(f"Shape: {df.shape}")
else:
    print("ERROR: Please place the Xente dataset in 'data/raw/' directory")
    exit(1)

# 1. Data Overview
print("\n" + "=" * 40)
print("1. DATA OVERVIEW")
print("=" * 40)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumns: {df.columns.tolist()}")

# 2. Missing Values
print("\n" + "=" * 40)
print("2. MISSING VALUES")
print("=" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing'] > 0])

# 3. Numerical Features
print("\n" + "=" * 40)
print("3. NUMERICAL FEATURES")
print("=" * 40)
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {list(numerical_cols)}")
print("\nSummary Statistics:")
print(df[numerical_cols].describe())

# 4. Categorical Features
print("\n" + "=" * 40)
print("4. CATEGORICAL FEATURES")
print("=" * 40)
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

for col in ['ProductCategory', 'ChannelId']:
    if col in df.columns:
        print(f"\n{col} value counts:")
        print(df[col].value_counts().head(5))

# 5. Time Analysis
print("\n" + "=" * 40)
print("5. TIME ANALYSIS")
print("=" * 40)
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['Hour'] = df['TransactionStartTime'].dt.hour
df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek

print(f"Date range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
print(f"Total days: {(df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days}")

# 6. Fraud Analysis
print("\n" + "=" * 40)
print("6. FRAUD ANALYSIS")
print("=" * 40)
if 'FraudResult' in df.columns:
    fraud_rate = df['FraudResult'].mean() * 100
    print(f"Fraud rate: {fraud_rate:.4f}%")
    print(f"Fraud transactions: {df['FraudResult'].sum():,}")

# 7. Customer Analysis
print("\n" + "=" * 40)
print("7. CUSTOMER ANALYSIS")
print("=" * 40)
customer_stats = df.groupby('CustomerId').agg({
    'Amount': ['count', 'sum', 'mean'],
    'TransactionId': 'count'
}).round(2)
customer_stats.columns = ['Transaction_Count', 'Total_Spend', 'Avg_Transaction', 'Num_Txns']
print(f"Unique customers: {len(customer_stats):,}")
print("\nCustomer Statistics:")
print(customer_stats.describe())

# 8. Key Insights
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM EDA")
print("=" * 60)
print("""
INSIGHT 1: Data Quality
- [Add your findings after running the analysis]

INSIGHT 2: Transaction Patterns  
- [Add your findings after running the analysis]

INSIGHT 3: Customer Behavior
- [Add your findings after running the analysis]

INSIGHT 4: Fraud Patterns
- [Add your findings after running the analysis]

INSIGHT 5: RFM Implications
- [Add your findings after running the analysis]
""")

print("\nEDA Complete!")
