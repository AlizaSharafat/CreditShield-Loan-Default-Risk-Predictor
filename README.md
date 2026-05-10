# CreditShield — Loan Default Risk Predictor

A machine learning application that predicts loan default risk using the [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) dataset from Kaggle. The system classifies applicants into **Low**, **Moderate**, or **High Risk** categories and provides explainable insights using SHAP.

## Live Demo

🔗 **[https://creditshield-loan-default-predictor.streamlit.app](https://creditshield-loan-default-predictor.streamlit.app)**

## Project Objective

Financial institutions need fast, consistent, and transparent credit risk assessment. This project builds an ML-powered system that accepts applicant financial data and returns a default-risk prediction with SHAP-based explanations showing why the model made that decision.

## Dataset

- **Source:** [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) (Kaggle)
- **Training file:** `cs-training.csv` (150,000 rows, 10 features)
- **Target variable:** `SeriousDlqin2yrs` — whether a borrower was 90+ days late within 2 years
- **Class balance:** 93.3% No Default / 6.7% Default

## Model

- **Algorithm:** XGBoost
- **Features:** 7 selected from 13 (10 original + 3 engineered)
- **Class imbalance:** SMOTE (ratio = 0.7)
- **AUC-ROC:** 0.85
- **Risk separation:** High Risk default rate = 36.4% vs Low Risk = 1.8% (20x difference)

## App Features

| Tab | Description |
|---|---|
| **Manual Input** | Enter 10 applicant fields, get risk prediction with SHAP explanation |
| **CSV Upload** | Upload a CSV file for batch predictions with downloadable results |
| **What-If Analysis** | Adjust sliders in real time to see how changes affect the risk score |

Every prediction includes a SHAP bar chart showing which features drove the decision and by how much.

## Repository Structure

```
CreditShield-Loan-Default-Risk-Predictor/
├── app/
│   ├── app.py                     # Streamlit web application
│   └── models/
│       ├── best_model.pkl
│       ├── best_threshold.pkl
│       ├── feature_cols.pkl
│       ├── final_scaler.pkl
│       ├── outlier_caps.pkl
│       └── train_medians.pkl
├── data_scripts/
│   ├── preprocess.py              # Missing values, duplicates, column mapping
│   ├── eda.py                     # EDA charts saved to figures/
│   ├── outliers_handling.py       # Outlier detection and capping
│   └── scaling_normalization.py   # Scaler comparison and RobustScaler
├── figures/                       # EDA plots
├── outputs/                       # Processed CSVs and pkl files
├── reports/                       # LaTeX source and compiled PDF
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Pipeline

1. **Preprocessing** — Column mapping, median imputation, duplicate removal
2. **EDA** — 7 charts (target distribution, histograms, correlation, boxplots, age/utilization analysis)
3. **Outlier handling** — Sentinel codes capped at 20, extreme values capped at 99th percentile
4. **Scaling** — RobustScaler (fitted on training set only)
5. **Feature engineering** — TotalLatePayments, HasSevereDelinquency, IncomeDebtRatio
6. **Feature selection** — Top 7 features by importance
7. **Model training** — XGBoost with SMOTE and threshold tuning
8. **Deployment** — Streamlit Cloud

## How to Run Locally

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```
