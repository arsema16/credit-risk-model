"""
Credit Risk Model - Exploratory Data Analysis
Task 2: Complete EDA for Xente eCommerce Transaction Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 150

print("=" * 80)
print("CREDIT RISK MODEL - EXPLORATORY DATA ANALYSIS")
print("Xente eCommerce Transaction Dataset")
print("=" * 80)

# Create figures directory if it doesn't exist
os.makedirs('notebooks/figures', exist_ok=True)

# ============================================================================
# 1. DATA LOADING WITH ERROR HANDLING
# ============================================================================

def load_data(data_path='data/raw/'):
    """Load the Xente dataset with error handling"""
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory {data_path} does not exist")
        
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        df = pd.read_csv(os.path.join(data_path, csv_files[0]))
        print(f"✓ Successfully loaded: {csv_files[0]}")
        print(f"  Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("  Please ensure the Xente dataset is in 'data/raw/' directory")
        return None

df = load_data()
if df is None:
    exit(1)

# ============================================================================
# 2. DATA OVERVIEW
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 1: DATA OVERVIEW")
print("=" * 80)

print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nData Types Summary:")
print(df.dtypes.value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

# ============================================================================
# 3. MISSING VALUES ANALYSIS WITH IMPUTATION STRATEGIES
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 2: MISSING VALUES ANALYSIS")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)

if len(missing_df) > 0:
    print("\nMissing Values Detected:")
    print(missing_df)
    
    print("\n📋 Proposed Imputation Strategies:")
    for col in missing_df.index:
        if df[col].dtype == 'object':
            print(f"  - {col}: Mode imputation (most frequent value)")
        else:
            print(f"  - {col}: Median imputation")
    
    print(f"\nTotal missing values: {missing.sum():,}")
    print(f"Total missing percentage: {(missing.sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
else:
    print("\n✓ No missing values found!")

# ============================================================================
# 4. NUMERICAL FEATURES - DISTRIBUTIONS AND STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: NUMERICAL FEATURES ANALYSIS")
print("=" * 80)

numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumerical columns: {list(numerical_cols)}")

print("\nSummary Statistics:")
print(df[numerical_cols].describe())

# FIGURE 1: Distribution of Numerical Features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

plot_cols = [col for col in ['Amount', 'Value', 'FraudResult'] if col in df.columns]

for idx, col in enumerate(plot_cols):
    df[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency (log scale)')
    axes[idx].set_yscale('log')
    axes[idx].grid(True, alpha=0.3)

if len(plot_cols) < 4:
    axes[3].set_visible(False)

plt.suptitle('Figure 1: Distribution of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/figures/figure1_numerical_distributions.png', bbox_inches='tight')
plt.close()
print("\n✓ Figure 1 saved: numerical_distributions.png")

# ============================================================================
# 5. OUTLIER DETECTION WITH BOX PLOTS AND IQR ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: OUTLIER DETECTION")
print("=" * 80)

if 'Amount' in df.columns:
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]
    
    print(f"\nAmount Feature Analysis:")
    print(f"  - Q1 (25th percentile): {Q1:.2f}")
    print(f"  - Q3 (75th percentile): {Q3:.2f}")
    print(f"  - IQR: {IQR:.2f}")
    print(f"  - Lower bound: {lower_bound:.2f}")
    print(f"  - Upper bound: {upper_bound:.2f}")
    print(f"  - Outliers detected: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  - Amount range: Min={df['Amount'].min():.2f}, Max={df['Amount'].max():.2f}")
    
    if len(outliers) / len(df) < 5:
        print("\n  📋 Recommended Action: Cap outliers at 99th percentile")
    else:
        print("\n  📋 Recommended Action: Consider separate modeling for high-value transactions")

# FIGURE 2: Box Plot
plt.figure(figsize=(10, 6))
df.boxplot(column='Amount', grid=True, patch_artist=True)
plt.title('Figure 2: Box Plot of Transaction Amount', fontsize=12, fontweight='bold')
plt.ylabel('Amount')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('notebooks/figures/figure2_amount_boxplot.png', bbox_inches='tight')
plt.close()
print("\n✓ Figure 2 saved: amount_boxplot.png")

# ============================================================================
# 6. CATEGORICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: CATEGORICAL FEATURES ANALYSIS")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

if 'ProductCategory' in df.columns:
    print("\nProduct Category Distribution (Top 10):")
    cat_counts = df['ProductCategory'].value_counts().head(10)
    print(cat_counts)
    
    plt.figure(figsize=(12, 6))
    cat_counts.plot(kind='bar', edgecolor='black')
    plt.title('Figure 3: Top 10 Product Categories', fontsize=12, fontweight='bold')
    plt.xlabel('Product Category')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('notebooks/figures/figure3_top_product_categories.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 saved: top_product_categories.png")

if 'ChannelId' in df.columns:
    print("\nChannel Distribution:")
    channel_counts = df['ChannelId'].value_counts()
    print(channel_counts)
    
    plt.figure(figsize=(10, 6))
    channel_counts.plot(kind='bar', edgecolor='black', color='lightblue')
    plt.title('Figure 4: Transaction Channels', fontsize=12, fontweight='bold')
    plt.xlabel('Channel')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('notebooks/figures/figure4_channel_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 saved: channel_distribution.png")

# ============================================================================
# 7. CORRELATION ANALYSIS WITH HEATMAP
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: CORRELATION ANALYSIS")
print("=" * 80)

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, linewidths=0.5)
plt.title('Figure 5: Correlation Matrix of Numerical Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/figures/figure5_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("\n✓ Figure 5 saved: correlation_heatmap.png")

if 'FraudResult' in corr_matrix.columns:
    fraud_corr = corr_matrix['FraudResult'].drop('FraudResult').sort_values(ascending=False)
    print("\nKey correlations with FraudResult:")
    for feature, corr_val in fraud_corr.head(3).items():
        print(f"  - {feature}: {corr_val:.3f}")

# ============================================================================
# 8. TIME-BASED ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: TIME-BASED ANALYSIS")
print("=" * 80)

df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['Hour'] = df['TransactionStartTime'].dt.hour
df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
df['Month'] = df['TransactionStartTime'].dt.month
df['Year'] = df['TransactionStartTime'].dt.year

print(f"\nDate range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
print(f"Total days: {(df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days}")

hourly_txns = df.groupby('Hour').size()
plt.figure(figsize=(14, 5))
hourly_txns.plot(kind='bar', edgecolor='black', color='steelblue')
plt.title('Figure 6: Transaction Volume by Hour of Day', fontsize=12, fontweight='bold')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/figures/figure6_hourly_volume.png', bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved: hourly_volume.png")

peak_hours = hourly_txns.nlargest(3).index.tolist()
print(f"\nPeak transaction hours: {peak_hours}")

daily_txns = df.groupby('DayOfWeek').size()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 5))
daily_txns.plot(kind='bar', edgecolor='black', color='coral')
plt.title('Figure 7: Transaction Volume by Day of Week', fontsize=12, fontweight='bold')
plt.xlabel('Day of Week')
plt.ylabel('Number of Transactions')
plt.xticks(range(7), days, rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/figures/figure7_weekly_pattern.png', bbox_inches='tight')
plt.close()
print("✓ Figure 7 saved: weekly_pattern.png")

# ============================================================================
# 9. CUSTOMER-LEVEL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: CUSTOMER-LEVEL ANALYSIS")
print("=" * 80)

customer_stats = df.groupby('CustomerId').agg({
    'Amount': ['count', 'sum', 'mean', 'std'],
    'TransactionId': 'count',
    'FraudResult': 'sum'
}).round(2)

customer_stats.columns = ['Transaction_Count', 'Total_Spend', 'Avg_Transaction', 
                          'Std_Transaction', 'Num_Txns', 'Fraud_Count']
customer_stats = customer_stats.fillna(0)

print(f"\nNumber of unique customers: {len(customer_stats):,}")
print("\nCustomer Transaction Statistics:")
print(customer_stats.describe())

customer_stats_sorted = customer_stats.sort_values('Total_Spend', ascending=False)
customer_stats_sorted['Cumulative_Spend_Pct'] = (
    customer_stats_sorted['Total_Spend'].cumsum() / 
    customer_stats_sorted['Total_Spend'].sum() * 100
)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(customer_stats_sorted) + 1), 
         customer_stats_sorted['Cumulative_Spend_Pct'], 
         linewidth=2, color='darkgreen')
plt.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
plt.xlabel('Customer Percentile', fontsize=11)
plt.ylabel('Cumulative Spend Percentage', fontsize=11)
plt.title('Figure 8: Customer Spend Concentration (Pareto Analysis)', 
          fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/figures/figure8_pareto_chart.png', bbox_inches='tight')
plt.close()
print("✓ Figure 8 saved: pareto_chart.png")

# ============================================================================
# 10. FRAUD ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: FRAUD ANALYSIS")
print("=" * 80)

if 'FraudResult' in df.columns:
    fraud_rate = df['FraudResult'].mean() * 100
    print(f"\nOverall fraud rate: {fraud_rate:.4f}%")
    print(f"Fraudulent transactions: {df['FraudResult'].sum():,}")
    print(f"Non-fraudulent transactions: {(df['FraudResult'] == 0).sum():,}")
    
    fraud_by_hour = df.groupby('Hour')['FraudResult'].mean() * 100
    plt.figure(figsize=(12, 5))
    fraud_by_hour.plot(kind='bar', edgecolor='black', color='darkred')
    plt.title('Figure 9: Fraud Rate by Hour of Day', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate (%)')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('notebooks/figures/figure9_fraud_by_hour.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 9 saved: fraud_by_hour.png")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 10: EDA SUMMARY")
print("=" * 80)

print("\n✅ EDA COMPLETE!")
print("📁 All figures saved to 'notebooks/figures/'")
print("\nKey Findings:")
print(f"  - Total transactions: {df.shape[0]:,}")
print(f"  - Unique customers: {len(customer_stats):,}")
print(f"  - Missing values: {missing.sum():,} ({missing.sum()/(df.shape[0]*df.shape[1])*100:.2f}%)")
if 'FraudResult' in df.columns:
    print(f"  - Fraud rate: {fraud_rate:.4f}%")
print(f"  - Peak transaction hours: {peak_hours}")
