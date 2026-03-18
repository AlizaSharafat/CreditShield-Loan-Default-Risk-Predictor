# CreditShield Loan Default Risk Predictor

CreditShield Loan Default Risk Predictor is a machine learning project that predicts borrower default risk using the Give Me Some Credit dataset(a competotion dataset downloaded from kaggle). The goal is to support faster and more consistent lending decisions by classifying applicants into risk categories based on their financial and credit behavior data.

## Project Objective

This project develops a default risk prediction system that analyzes applicant financial and credit-related features to estimate the likelihood of serious delinquency. The model output can later be mapped into Low, Medium, and High risk categories for easier interpretation and decision-making.

## Dataset

This project uses the **Give Me Some Credit** dataset.

- **Training file:** `cs-training.csv`
- **Target variable:** `SeriousDlqin2yrs`

The target variable indicates whether a borrower experienced serious delinquency (90 days past due or worse) within two years.

## Current Deliverable Scope

This repository currently includes work for:

- Problem definition
- Dataset understanding
- Missing value analysis
- Duplicate detection and removal
- Initial preprocessing
- Report preparation
- Outlier handling
- Feature scaling and normalization

Further stages will include:

- Model training and evaluation
- Class imbalance handling using methods such as SMOTE
- Explainable AI using SHAP/LIME
- What-if analysis in the final application UI

## Repository Structure

```text
CreditShield-Loan-Default-Risk-Predictor/
│
├── data_scripts/     # Python scripts for preprocessing and analysis
├── figures/          # Saved charts and figures for the report
├── reports/          # LaTeX source and final report files
├── .gitignore        # Files and folders ignored by Git
├── README.md         # Project overview
└── requirements.txt  # Python dependencies
