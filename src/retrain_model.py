"""
Simple retraining script to save model in compatible format
No MLflow dependency - just trains and saves the model
"""

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 60)
print("RETRAINING MODEL FOR COMPATIBILITY")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/processed/processed_data.csv')

# Prepare features
exclude_cols = ['CustomerId', 'is_high_risk', 'Recency', 'Frequency', 'Monetary', 'Cluster']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['is_high_risk']

print(f"    Features shape: {X.shape}")
print(f"    Target distribution: {y.value_counts().to_dict()}")

# Split data
print("\n[2] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train size: {X_train.shape[0]}")
print(f"    Test size: {X_test.shape[0]}")

# Train model
print("\n[3] Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    subsample=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("\n[4] Evaluating model...")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"    Train accuracy: {train_score:.4f}")
print(f"    Test accuracy: {test_score:.4f}")

# Save model
print("\n[5] Saving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/best_model.pkl')
print("    ✓ Model saved to: models/best_model.pkl")

# Feature importance
print("\n[6] Feature importance:")
importances = model.feature_importances_
imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False)
print(imp_df.head(10).to_string(index=False))

# Save feature importance to CSV
os.makedirs('docs/images', exist_ok=True)
imp_df.to_csv('docs/images/feature_importance.csv', index=False)
print("    ✓ Feature importance saved to: docs/images/feature_importance.csv")

print("\n" + "=" * 60)
print("RETRAINING COMPLETE!")
print("=" * 60)
