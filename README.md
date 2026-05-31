# credit-risk-model
Credit Risk Probability Model for Alternative Data - Bati Bank BNPL Service
# Credit Risk Probability Model for Alternative Data

## Project Overview
This project builds an end-to-end credit scoring system for Bati Bank's buy-now-pay-later (BNPL) service using eCommerce transaction data.

## Timeline
- **Challenge Start**: 28 May 2026
- **Interim Submission**: 31 May 2026 (8:00 PM UTC)
- **Final Submission**: 03 June 2026 (8:00 PM UTC)

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

**Why a proxy is necessary**: Without a target variable, supervised machine learning is impossible. The proxy allows us to create a "synthetic ground truth" by identifying behavioral patterns that correlate with credit risk (disengaged customers → higher default probability).

**Business Risks of Proxy-Based Prediction**:

| Risk | Description | Mitigation Strategy |
|------|-------------|---------------------|
| Label Bias | High-risk proxy may not equal actual default | Validate with expert judgment from bank's risk team |
| Concept Drift | Customer behavior patterns change over time | Implement regular model retraining (quarterly) |
| Causal Confusion | Correlation between disengagement and default ≠ causation | Use feature importance analysis to validate relationships |
| Regulatory Scrutiny | Proxy methodology must be justified to regulators | Comprehensive documentation of RFM clustering approach |

### Key Trade-offs: Interpretable vs. High-Performance Models

| Dimension | Simple Interpretable Model (Logistic Regression + WoE) | High-Performance Model (Gradient Boosting) |
|-----------|--------------------------------------------------------|---------------------------------------------|
| **Interpretability** | Excellent - coefficients directly show marginal effects and direction of impact | Poor - feature interactions obscure individual contributions; requires SHAP/LIME |
| **Regulatory Acceptance** | High - industry standard for credit scoring, well-understood by regulators | Medium - requires additional explanation tools and justification |
| **Predictive Power** | Good for linear relationships; may miss complex patterns | Excellent for non-linear patterns and feature interactions |
| **Maintenance Cost** | Low - stable coefficients, easy to monitor | High - prone to overfitting, needs frequent retraining |
| **Implementation Speed** | Fast - minimal tuning required | Moderate - requires hyperparameter optimization |

**Decision Framework for This Project**:
- Train both model types and compare performance
- If Gradient Boosting improves ROC-AUC by <5%, select Logistic Regression for better regulatory compliance
- If Gradient Boosting shows >5% improvement, select it but provide SHAP explanations
---
## Repository Structure (To be completed in subsequent tasks)
credit-risk-model/
├── .github/workflows/ # CI/CD pipeline (Task 6)
├── data/ # Data directory (Task 2-4)
├── notebooks/ # EDA (Task 2)
├── src/ # Source code (Task 3-6)
├── tests/ # Unit tests (Task 5)
├── Dockerfile # Containerization (Task 6)
└── requirements.txt # Dependencies (Task 2)
