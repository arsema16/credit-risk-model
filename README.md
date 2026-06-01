# Credit Risk Probability Model for Alternative Data

## Project Overview
This project builds an end-to-end credit scoring system for Bati Bank's buy-now-pay-later (BNPL) service using eCommerce transaction data. The model transforms behavioral data into predictive risk signals and deploys them as a REST API.

## Team Members
- Kerod
- Mahbubah
- Feven

## Timeline
- **Challenge Start**: 28 May 2026
- **Interim Submission**: 31 May 2026 (8:00 PM UTC)
- **Final Submission**: 03 June 2026 (8:00 PM UTC)

---

## Data Source

**Dataset:** Xente eCommerce Transaction Dataset

**Source:** Kaggle - Xente Challenge
- URL: https://www.kaggle.com/competitions/xente-challenge

**Dataset Description:**
This dataset contains transaction-level records from the Xente eCommerce platform. Each row represents a single customer transaction with the following fields:

| Field | Description |
|-------|-------------|
| TransactionId | Unique transaction identifier |
| AccountId | Unique account identifier |
| CustomerId | Unique customer identifier |
| Amount | Transaction value (positive=debit, negative=credit) |
| Value | Absolute value of Amount |
| TransactionStartTime | Transaction timestamp |
| ProductCategory | Product category grouping |
| ChannelId | Customer channel (web, Android, iOS, pay-later, checkout) |
| PricingStrategy | Xente pricing structure category |
| FraudResult | Fraud flag (1=fraud, 0=no fraud) |

**Important Usage Notes:**
- This dataset contains NO historical default labels
- The fraud flag indicates transaction fraud, not credit default
- Credit risk proxy must be engineered using behavioral patterns (RFM analysis)
- Data is suitable for customer segmentation and feature engineering
- **Limitations:** No demographic data, no external credit history, no prior loan performance data

---

## Credit Scoring Business Understanding

### How the Basel II Accord's Emphasis on Risk Measurement Influences Model Design

The Basel II Capital Accord establishes three pillars for banking regulation, with Pillar 1 focusing on minimum capital requirements for credit risk. Under the Internal Ratings-Based (IRB) approach, banks must demonstrate that their models are:

1. **Statistically Sound**: Models must provide "meaningful differentiation of risk" and accurate quantification of risk parameters (Probability of Default, Loss Given Default, Exposure at Default).

2. **Well-Documented**: Every modeling decision—from variable selection to validation methodology—must be fully documented and traceable for regulatory review.

3. **Interpretable**: Regulators and internal risk committees must understand how the model arrives at its predictions. Black-box models face significant scrutiny.

4. **Subject to Governance**: Models require regular validation, performance monitoring, and approval before deployment.

### Why a Proxy Variable is Necessary Without a Direct Default Label

The raw transaction data contains no historical default flag because:
- The eCommerce platform has never offered credit before
- No prior loan performance data exists
- Only behavioral transaction patterns are available

**Why a proxy is necessary**: Without a target variable, supervised machine learning is impossible. The proxy allows us to create a "synthetic ground truth" by identifying behavioral patterns that correlate with credit risk.

**Business Risks of Proxy-Based Prediction**:

| Risk | Description | Mitigation Strategy |
|------|-------------|---------------------|
| Label Bias | High-risk proxy may not equal actual default | Validate with expert judgment from bank's risk team |
| Concept Drift | Customer behavior patterns change over time | Implement regular model retraining (quarterly) |
| Causal Confusion | Correlation between disengagement and default ≠ causation | Use feature importance analysis to validate relationships |
| Regulatory Scrutiny | Proxy methodology must be justified to regulators | Comprehensive documentation of RFM clustering approach |

### Key Trade-offs: Interpretable vs. High-Performance Models

| Dimension | Simple Interpretable Model (Logistic Regression) | High-Performance Model (Gradient Boosting) |
|-----------|--------------------------------------------------------|---------------------------------------------|
| **Interpretability** | Excellent - coefficients directly show marginal effects | Poor - requires SHAP/LIME for explanation |
| **Regulatory Acceptance** | High - industry standard for credit scoring | Medium - requires additional justification |
| **Predictive Power** | Good for linear relationships | Excellent for non-linear patterns |
| **Maintenance Cost** | Low - stable coefficients | High - needs frequent retraining |
| **Implementation Speed** | Fast - minimal tuning required | Moderate - requires hyperparameter optimization |

**Decision Framework for This Project**:
- Train both model types and compare performance
- If Gradient Boosting improves ROC-AUC by <5%, select Logistic Regression for better regulatory compliance
- If Gradient Boosting shows >5% improvement, select it but provide SHAP explanations

---

## Repository Structure
```
credit-risk-model/
├── .github/workflows/ci.yml # CI/CD pipeline (Task 6)
├── data/
│ ├── raw/ # Raw data (gitignored)
│ └── processed/ # Processed data (gitignored)
├── notebooks/
│ └── eda_complete.py # Exploratory data analysis (Task 2)
├── src/
│ ├── init.py
│ ├── data_processing.py # Feature engineering (Task 3)
│ └── api/
│ ├── main.py # FastAPI application (Task 6)
│ └── pydantic_models.py # Request/response schemas (Task 6)
├── tests/
│ └── test_data_processing.py # Unit tests (Task 5)
├── models/ # Saved models (gitignored)
├── mlruns/ # MLflow tracking (gitignored)
├── Dockerfile # Containerization (Task 6)
├── docker-compose.yml # Multi-service orchestration (Task 6)
├── requirements.txt # Dependencies (Task 2)
├── .gitignore # Git ignore rules (Task 2)
└── README.md # This file (Task 1)
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/arsema16/credit-risk-model.git
cd credit-risk-model
```
### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download Dataset
Download the Xente Challenge dataset from Kaggle and place it in data/raw/

### 5. Run EDA
```bash
python notebooks/eda_complete.py
```
### 6. Run Feature Engineering
```bash
python src/data_processing.py
```
### 7. Train Models (Task 5)
```bash
python src/train.py
```
### 8. Start MLflow UI
```bash
mlflow ui
```
Navigate to http://localhost:5000