# ==========================================
# Import Libraries
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv(
    "dashboard_data.csv"
)

model = joblib.load(
    "model.pkl"
)

explainer = shap.Explainer(
    model
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title(
    "Fraud Detection Dashboard"
)

# Sidebar Risk Filter

selected_risk = st.sidebar.multiselect(
    "Filter By Risk Tier",
    options=df["RiskTier"].unique(),
    default=df["RiskTier"].unique()
)

filtered_df = df[
    df["RiskTier"].isin(selected_risk)
]

# Sidebar Navigation

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer"
    ]
)

# ==========================================
# PAGE 1 — OVERVIEW
# ==========================================

if page == "Overview":

    st.title(
        "Fraud Detection Overview"
    )

    # Metrics

    total_transactions = len(
        filtered_df
    )

    total_fraud = filtered_df[
        "ActualFraud"
    ].sum()

    detection_rate = (
        total_fraud /
        total_transactions
    ) * 100

    avg_fraud_amount = filtered_df[
        filtered_df["ActualFraud"] == 1
    ]["TransactionAmt"].mean()

    # Layout

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Total Fraud Cases",
        int(total_fraud)
    )

    col1.metric(
        "Detection Rate",
        f"{detection_rate:.2f}%"
    )

    col2.metric(
        "Average Fraud Amount",
        f"${avg_fraud_amount:.2f}"
    )

    # ======================================
    # Risk Tier Distribution
    # ======================================

    st.subheader(
        "Risk Tier Distribution"
    )

    risk_chart = px.histogram(
        filtered_df,
        x="RiskTier",
        color="RiskTier",
        title="Risk Tier Distribution"
    )

    st.plotly_chart(
        risk_chart,
        use_container_width=True
    )

    # ======================================
    # Fraud Probability Distribution
    # ======================================

    st.subheader(
        "Fraud Probability Distribution"
    )

    prob_chart = px.histogram(
        filtered_df,
        x="FraudProbability",
        color="RiskTier",
        nbins=30
    )

    st.plotly_chart(
        prob_chart,
        use_container_width=True
    )

# ==========================================
# PAGE 2 — TRANSACTION EXPLORER
# ==========================================

elif page == "Transaction Explorer":

    st.title(
        "Transaction Explorer"
    )

    # Search TransactionID

    transaction_id = st.number_input(
        "Enter TransactionID",
        min_value=0
    )

    transaction_data = filtered_df[
        filtered_df["TransactionID"]
        == transaction_id
    ]

    # Display Result

    if len(transaction_data) > 0:

        st.subheader(
            "Transaction Details"
        )

        st.write(
            transaction_data
        )

        # Risk Score

        risk_score = transaction_data[
            "FraudProbability"
        ].values[0]

        st.metric(
            "Fraud Probability",
            f"{risk_score:.4f}"
        )

        # Risk Messages

        if risk_score >= 0.75:

            st.error(
                "Critical Risk Transaction"
            )

        elif risk_score >= 0.40:

            st.warning(
                "Suspicious Transaction"
            )

        else:

            st.success(
                "Clear Transaction"
            )

    # ======================================
    # Searchable Table
    # ======================================

    st.subheader(
        "Searchable Transaction Table"
    )

    st.dataframe(
        filtered_df.head(100)
    )

# ==========================================
# PAGE 3 — SHAP EXPLAINER
# ==========================================

elif page == "SHAP Explainer":

    st.title(
        "SHAP Fraud Explainer"
    )

    shap_transaction_id = st.number_input(
        "Enter TransactionID",
        min_value=0,
        key="shap_input"
    )

    shap_df = filtered_df[
        filtered_df["TransactionID"]
        == shap_transaction_id
    ]

    if len(shap_df) > 0:

        st.subheader(
            "Transaction Details"
        )

        st.write(
            shap_df
        )

        # ======================================
        # Feature Selection
        # ======================================

        feature_df = shap_df.drop(
            columns=[
                "ActualFraud",
                "FraudProbability",
                "RiskTier"
            ],
            errors="ignore"
        )

        # ======================================
        # Generate SHAP Values
        # ======================================

        shap_values = explainer(
            feature_df
        )

        # ======================================
        # SHAP Waterfall Plot
        # ======================================

        st.subheader(
            "SHAP Waterfall Plot"
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(fig)

        # ======================================
        # Plain English Explanation
        # ======================================

        st.subheader(
            "Plain English Explanation"
        )

        probability = shap_df[
            "FraudProbability"
        ].values[0]

        if probability >= 0.75:

            st.error(
                f"""
                HIGH FRAUD RISK
                
                Fraud probability is {probability:.2%}.
                
                The model detected multiple
                high-risk behavioral signals.
                
                Immediate investigation is recommended.
                """
            )

        elif probability >= 0.40:

            st.warning(
                f"""
                SUSPICIOUS TRANSACTION
                
                Fraud probability is {probability:.2%}.
                
                Some features partially match
                historical fraud patterns.
                """
            )

        else:

            st.success(
                f"""
                LEGITIMATE TRANSACTION
                
                Fraud probability is {probability:.2%}.
                
                Transaction characteristics appear normal.
                """
            )