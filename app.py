"""
Streamlit Dashboard - Credit Risk Model
Minimal dashboard for Week 12 submission
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("🏦 Bati Bank – Credit Risk Predictor")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists('models/best_model.pkl'):
        return joblib.load('models/best_model.pkl')
    return None

model = load_model()

if model:
    st.success("✅ Model loaded successfully")
else:
    st.error("❌ Model not found. Run src/retrain_model.py first")

st.markdown("---")

# Tab 1: Portfolio Overview
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🔍 Predict Risk", "📈 Model Performance"])

with tab1:
    st.write("### Portfolio Overview")
    
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            high_risk = df['is_high_risk'].sum()
            st.metric("High-Risk Customers", f"{high_risk:,}")
        with col3:
            risk_pct = df['is_high_risk'].mean() * 100
            st.metric("High-Risk Rate", f"{risk_pct:.1f}%")
        
        # Risk distribution chart
        fig, ax = plt.subplots()
        df['is_high_risk'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title('Risk Distribution (0 = Low Risk, 1 = High Risk)')
        ax.set_xlabel('Risk Category')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
    except:
        st.warning("Data not found. Place processed data in data/processed/")

with tab2:
    st.write("### Predict Risk")
    st.write("Enter customer transaction details to get a risk prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        total_amount = st.number_input("Total Amount Spent", min_value=0.0, value=500.0)
        avg_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=50.0)
        transaction_count = st.number_input("Number of Transactions", min_value=1, value=5)
    
    with col2:
        recency = st.number_input("Days Since Last Transaction", min_value=0, value=30)
        fraud_rate = st.slider("Fraud Rate (%)", 0.0, 100.0, 5.0) / 100
        customer_lifespan = st.number_input("Customer Lifespan (days)", min_value=0, value=90)
    
    if st.button("Predict Risk"):
        if model:
            # Create feature vector (simplified)
            features = pd.DataFrame([{
                'total_amount': total_amount,
                'avg_amount': avg_amount,
                'transaction_count': transaction_count,
                'fraud_rate': fraud_rate,
                'customer_lifespan_days': customer_lifespan,
                # Add other features with default values
                'std_amount': 0,
                'min_amount': 0,
                'max_amount': 0,
                'fraud_count': 0,
                'unique_product_categories': 1,
                'unique_channels': 1,
                'avg_amount_per_transaction': avg_amount,
                'monetary_variability': 0,
                'log_total_amount': np.log1p(total_amount),
                'log_avg_amount': np.log1p(avg_amount),
                'avg_transaction_hour': 12,
                'night_transaction_ratio': 0,
                'weekend_transaction_ratio': 0
            }])
            
            # Ensure all features are present
            if hasattr(model, 'feature_importances_'):
                # Get feature names from model (approximate)
                prob = model.predict_proba(features)[0][1]
                st.write(f"### Risk Probability: {prob:.2%}")
                
                if prob < 0.3:
                    st.success("✅ Low Risk - Auto-approve")
                    st.write("Recommended Credit Limit: $10,000")
                    st.write("Recommended Duration: 12 months")
                elif prob < 0.6:
                    st.warning("⚠️ Medium Risk - Manual Review")
                    st.write("Recommended Credit Limit: $5,000")
                    st.write("Recommended Duration: 9 months")
                else:
                    st.error("❌ High Risk - Consider Declining")
                    st.write("Recommended Credit Limit: $2,000")
                    st.write("Recommended Duration: 6 months")

with tab3:
    st.write("### Model Performance")
    
    try:
        # Load feature importance if available
        if os.path.exists('docs/images/feature_importance.csv'):
            imp_df = pd.read_csv('docs/images/feature_importance.csv')
            top_features = imp_df.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Most Important Features')
            st.pyplot(fig)
        else:
            st.info("Run src/retrain_model.py to generate feature importance data")
    except:
        st.warning("Feature importance data not available")

st.markdown("---")
st.caption("Built for 10 Academy Week 12 Challenge")
