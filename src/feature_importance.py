"""
Feature Importance Analysis (Simplified - no SHAP dependency)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('docs/images', exist_ok=True)

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Load model
print("\n[1] Loading model...")
model = joblib.load('models/best_model.pkl')
print(f"    Model type: {type(model).__name__}")

# Load data
print("\n[2] Loading data...")
df = pd.read_csv('data/processed/processed_data.csv')
exclude_cols = ['CustomerId', 'is_high_risk', 'Recency', 'Frequency', 'Monetary', 'Cluster']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
print(f"    Features: {len(feature_cols)} columns")

# Get feature importance
print("\n[3] Extracting feature importance...")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    method = "Gini Importance"
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_[0])
    method = "Coefficient Magnitude"
else:
    print("    Model doesn't support feature importance")
    importances = np.ones(len(feature_cols))
    method = "Uniform"

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\n    Top 10 features ({method}):")
print(importance_df.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
top_features = importance_df.head(10)
colors = plt.cm.Blues_r(np.linspace(0.4, 0.9, len(top_features)))[::-1]
plt.barh(top_features['feature'], top_features['importance'], color=colors)
plt.xlabel(f'Importance ({method})', fontsize=12)
plt.title(f'Top 10 Most Important Features\n{method}', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('docs/images/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n    ✓ Saved: docs/images/feature_importance.png")

# Save CSV
importance_df.to_csv('docs/images/feature_importance.csv', index=False)
print("    ✓ Saved: docs/images/feature_importance.csv")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE COMPLETE!")
print("=" * 60)
