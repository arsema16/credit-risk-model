# Credit Risk Probability Model for Alternative Data

## Project Overview
This project builds an end-to-end credit scoring system for Bati Bank's buy-now-pay-later (BNPL) service using eCommerce transaction data.



## Credit Scoring Business Understanding

### How the Basel II Accord's Emphasis on Risk Measurement Influences Model Design

The Basel II Capital Accord establishes three pillars for banking regulation, with Pillar 1 focusing on minimum capital requirements for credit risk. Under the Internal Ratings-Based (IRB) approach, banks must demonstrate that their models are:

1. **Statistically Sound**: Models must provide "meaningful differentiation of risk" and accurate quantification of risk parameters.

2. **Well-Documented**: Every modeling decision must be fully documented and traceable for regulatory review.

3. **Interpretable**: Regulators and internal risk committees must understand how the model arrives at its predictions.

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
