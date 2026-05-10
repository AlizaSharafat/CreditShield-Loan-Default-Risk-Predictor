import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CreditShield", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    artifacts = {}
    for name in ["best_model", "final_scaler", "feature_cols", "best_threshold", "outlier_caps", "train_medians"]:
        with open(os.path.join(BASE_DIR, "models", f"{name}.pkl"), "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts["best_model"]
scaler = artifacts["final_scaler"]
feature_cols = artifacts["feature_cols"]
threshold = artifacts["best_threshold"]
caps = artifacts["outlier_caps"]
medians = artifacts["train_medians"]


def preprocess(df):
    """Apply the same pipeline used during training."""
    df = df.copy()

    # Fill missing values with training medians
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(medians["MonthlyIncome"])
    df["Dependents"] = df["Dependents"].fillna(medians["Dependents"])
    if (df["Age"] == 0).any():
        df.loc[df["Age"] == 0, "Age"] = medians["Age"]

    # Cap sentinel values in late payment columns
    for col in ["Times30_59Late", "Times60_89Late", "Times90Late"]:
        if col in df.columns:
            df.loc[df[col] >= 96, col] = 20

    # Cap outliers using training percentiles
    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    # Feature engineering
    df["TotalLatePayments"] = df["Times30_59Late"] + df["Times60_89Late"] + df["Times90Late"]
    df["HasSevereDelinquency"] = (df["Times90Late"] > 0).astype(int)
    df["IncomeDebtRatio"] = df["MonthlyIncome"] / (df["DebtRatio"] + 1)
    if "IncomeDebtRatio" in caps:
        df["IncomeDebtRatio"] = df["IncomeDebtRatio"].clip(upper=caps["IncomeDebtRatio"])

    # Select and scale
    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
    return X_scaled, df

def predict(X_scaled):
    """Get predictions, probabilities, and risk categories."""
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    risks = []
    for p in probs:
        if p < 0.2:
            risks.append("Low Risk")
        elif p < 0.5:
            risks.append("Moderate Risk")
        else:
            risks.append("High Risk")
    return preds, probs, risks

def risk_color(risk):
    if risk == "Low Risk":
        return "#2ecc71"
    elif risk == "Moderate Risk":
        return "#f39c12"
    else:
        return "#e74c3c"

# ── SHAP Explainer 
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# ── Header 
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1>🛡️ CreditShield</h1>
    <p style='font-size: 1.1rem; color: gray;'>Loan Default Risk Predictor with Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs 
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📄 CSV Upload", "🔮 What-If Analysis"])

# TAB 1: Manual Input

with tab1:
    st.subheader("Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=35)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5000)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.3, step=0.01)

    with col2:
        revolving_util = st.number_input("Revolving Utilization", min_value=0.0, max_value=2.0, value=0.15, step=0.01,
                                          help="Credit card balance / credit limit. Values above 1.0 mean overlimit.")
        open_credit = st.number_input("Open Credit Lines", min_value=0, max_value=30, value=8)
        real_estate = st.number_input("Real Estate Lines", min_value=0, max_value=10, value=1)

    with col3:
        times_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)
        times_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)
        times_90 = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Build dataframe from inputs
        input_data = pd.DataFrame([{
            "RevolvingUtilization": revolving_util, "Age": age,
            "Times30_59Late": times_30_59, "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income, "OpenCreditLines": open_credit,
            "Times90Late": times_90, "RealEstateLines": real_estate,
            "Times60_89Late": times_60_89, "Dependents": dependents,
        }])

        X_scaled, _ = preprocess(input_data)
        preds, probs, risks = predict(X_scaled)

        prob = probs[0]
        risk = risks[0]
        color = risk_color(risk)

        # Display result
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Category", risk)
        r2.metric("Default Probability", f"{prob:.1%}")
        r3.metric("Prediction", "Default" if preds[0] == 1 else "No Default")

        st.markdown(f"""
        <div style='background-color: {color}22; border-left: 5px solid {color}; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
            <h3 style='color: {color}; margin: 0;'>{risk}</h3>
            <p style='margin: 0.5rem 0 0 0;'>This applicant has a <b>{prob:.1%}</b> probability of defaulting within 2 years.</p>
        </div>
        """, unsafe_allow_html=True)

        # SHAP explanation
        st.subheader("🔎 Why This Prediction? (SHAP Explanation)")
        shap_values = explainer.shap_values(X_scaled)
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": X_scaled.iloc[0].values,
            "SHAP Impact": shap_values[0]
        }).sort_values("SHAP Impact", key=abs, ascending=False)

        # Waterfall-style bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors_shap = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Impact"]]
        ax.barh(shap_df["Feature"], shap_df["SHAP Impact"], color=colors_shap, edgecolor="black", alpha=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title("Feature Contributions to This Prediction", fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🔴 Red bars push toward Default | 🟢 Green bars push toward No Default")

        # Text explanation
        top_features = shap_df.head(3)
        reasons = []
        for _, row in top_features.iterrows():
            direction = "increases" if row["SHAP Impact"] > 0 else "decreases"
            reasons.append(f"**{row['Feature']}** {direction} the risk")
        st.markdown("**Top factors:** " + " | ".join(reasons))


# TAB 2: CSV Upload

with tab2:
    st.subheader("Upload Applicant Data (CSV)")

    st.markdown("""
    Upload a CSV file with these columns:
    `RevolvingUtilization`, `Age`, `Times30_59Late`, `DebtRatio`, `MonthlyIncome`,
    `OpenCreditLines`, `Times90Late`, `RealEstateLines`, `Times60_89Late`, `Dependents`
    """)

    # Also accept original Kaggle column names
    kaggle_map = {
        "SeriousDlqin2yrs": "Target",
        "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilization",
        "age": "Age",
        "NumberOfTime30-59DaysPastDueNotWorse": "Times30_59Late",
        "NumberOfTime60-89DaysPastDueNotWorse": "Times60_89Late",
        "NumberOfTimes90DaysLate": "Times90Late",
        "NumberOfOpenCreditLinesAndLoans": "OpenCreditLines",
        "NumberRealEstateLoansOrLines": "RealEstateLines",
        "NumberOfDependents": "Dependents",
    }

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        # Drop unnamed index if present
        if "Unnamed: 0" in df_upload.columns:
            df_upload = df_upload.drop(columns=["Unnamed: 0"])

        # Auto-detect and rename Kaggle columns
        renamed = False
        for old, new in kaggle_map.items():
            if old in df_upload.columns:
                df_upload = df_upload.rename(columns={old: new})
                renamed = True
        if renamed:
            st.info("Detected original Kaggle column names — automatically renamed.")

        st.write(f"Uploaded {df_upload.shape[0]:,} rows × {df_upload.shape[1]} columns")
        st.dataframe(df_upload.head(), use_container_width=True)

        if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                X_scaled, df_processed = preprocess(df_upload)
                preds, probs, risks = predict(X_scaled)

                # Add results to dataframe
                results = df_upload.copy()
                results["DefaultProbability"] = np.round(probs, 4)
                results["RiskCategory"] = risks
                results["Prediction"] = ["Default" if p == 1 else "No Default" for p in preds]

            # Summary
            st.markdown("---")
            st.subheader("Results Summary")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Applicants", f"{len(results):,}")
            s2.metric("Low Risk", f"{risks.count('Low Risk'):,}")
            s3.metric("Moderate Risk", f"{risks.count('Moderate Risk'):,}")
            s4.metric("High Risk", f"{risks.count('High Risk'):,}")

            # Risk distribution chart
            fig, ax = plt.subplots(figsize=(6, 2.5))
            risk_counts = pd.Series(risks).value_counts().reindex(["Low Risk", "Moderate Risk", "High Risk"], fill_value=0)
            colors_bar = ["#2ecc71", "#f39c12", "#e74c3c"]
            ax.bar(risk_counts.index, risk_counts.values, color=colors_bar, edgecolor="black")
            for i, v in enumerate(risk_counts.values):
                ax.text(i, v + max(risk_counts.values)*0.02, f"{v:,}", ha="center", fontweight="bold")
            ax.set_title("Risk Distribution", fontweight="bold")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Results table
            st.subheader("Detailed Results")
            st.dataframe(results, use_container_width=True)

            # Download
            csv_out = results.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv_out, "predictions.csv", "text/csv",
                               use_container_width=True)

            # SHAP for first row as example
            st.subheader("🔎 SHAP Explanation (First Applicant)")
            shap_values = explainer.shap_values(X_scaled.iloc[[0]])
            shap_df = pd.DataFrame({
                "Feature": feature_cols,
                "SHAP Impact": shap_values[0]
            }).sort_values("SHAP Impact", key=abs, ascending=False)

            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors_shap = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Impact"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Impact"], color=colors_shap, edgecolor="black", alpha=0.8)
            ax.set_xlabel("SHAP Value")
            ax.set_title("Feature Contributions (First Applicant)", fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("🔴 Red = pushes toward Default | 🟢 Green = pushes toward No Default")


# TAB 3: What-If Analysis

with tab3:
    st.subheader("What-If Analysis")
    st.markdown("Adjust the sliders to see how changes affect the risk score in real time.")

    wc1, wc2 = st.columns([1, 1])

    with wc1:
        wi_age = st.slider("Age", 18, 100, 35)
        wi_income = st.slider("Monthly Income ($)", 0, 30000, 5000, step=500)
        wi_debt = st.slider("Debt Ratio", 0.0, 5.0, 0.3, step=0.05)
        wi_util = st.slider("Revolving Utilization", 0.0, 1.5, 0.15, step=0.05)
        wi_open = st.slider("Open Credit Lines", 0, 25, 8)

    with wc2:
        wi_real = st.slider("Real Estate Lines", 0, 5, 1)
        wi_dep = st.slider("Dependents", 0, 10, 0)
        wi_30 = st.slider("Times 30-59 Days Late", 0, 20, 0)
        wi_60 = st.slider("Times 60-89 Days Late", 0, 20, 0)
        wi_90 = st.slider("Times 90+ Days Late", 0, 20, 0)

    # Build input and predict
    wi_data = pd.DataFrame([{
        "RevolvingUtilization": wi_util, "Age": wi_age,
        "Times30_59Late": wi_30, "DebtRatio": wi_debt,
        "MonthlyIncome": wi_income, "OpenCreditLines": wi_open,
        "Times90Late": wi_90, "RealEstateLines": wi_real,
        "Times60_89Late": wi_60, "Dependents": wi_dep,
    }])

    X_wi_scaled, _ = preprocess(wi_data)
    _, wi_probs, wi_risks = predict(X_wi_scaled)
    wi_prob = wi_probs[0]
    wi_risk = wi_risks[0]
    wi_color = risk_color(wi_risk)

    # Display
    st.markdown("---")

    # Risk gauge
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem; background-color: {wi_color}22;
                border: 2px solid {wi_color}; border-radius: 10px; margin: 1rem 0;'>
        <h2 style='color: {wi_color}; margin: 0;'>{wi_risk}</h2>
        <h1 style='margin: 0.5rem 0;'>{wi_prob:.1%}</h1>
        <p style='color: gray; margin: 0;'>Default Probability</p>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar visual
    st.progress(float(min(wi_prob, 1.0)))

    # SHAP for this scenario
    shap_values = explainer.shap_values(X_wi_scaled)
    shap_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Impact": shap_values[0]
    }).sort_values("SHAP Impact", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    colors_shap = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Impact"]]
    ax.barh(shap_df["Feature"], shap_df["SHAP Impact"], color=colors_shap, edgecolor="black", alpha=0.8)
    ax.set_xlabel("SHAP Value")
    ax.set_title("What's Driving This Risk Score?", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Move the sliders above to see how each change affects the prediction in real time.")

# ── Footer 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85rem;'>
    CreditShield — Loan Default Risk Predictor | Built with XGBoost + SHAP<br>
    23L-2524 & 23L-2560
</div>
""", unsafe_allow_html=True)