# 🚨 Fraud Detection System using Machine Learning & Explainable AI

An advanced end-to-end Fraud Detection System built using Machine Learning, SHAP Explainable AI, Risk Segmentation, and an interactive Streamlit Dashboard.

The project focuses on detecting fraudulent financial transactions using supervised learning models and explainable AI techniques to improve fraud monitoring and business decision-making.

---

# 📌 Project Features

✅ Data Cleaning & Preprocessing  
✅ Feature Engineering  
✅ Handling Imbalanced Data using SMOTE  
✅ Multiple ML Model Training & Comparison  
✅ SHAP Explainable AI Integration  
✅ Fraud Risk Segmentation  
✅ Interactive Streamlit Dashboard  
✅ Fraud Analytics Visualizations  
✅ Threshold Optimization  
✅ Business Recommendations  

---

# 📂 Project Structure

```bash
FraudDetection_Saransh_Malik_Batch_C/
│
├── analysis.ipynb
├── README.md
├── requirements.txt
├── summary.docx
├── shap_summary.png
├── model_comparison.png
│
├── data/
│   ├── train_transaction.csv
│   └── train_identity.csv
│
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── dashboard_data.csv
│
├── charts/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── fraud_rate_by_hour.png
│   ├── transaction_amount_distribution.png
│   ├── risk_tier_donut_chart.png
│   └── other generated charts
```

---

# 📊 Dataset Information

The project uses two datasets:

- `train_transaction.csv`
- `train_identity.csv`

Both datasets were merged using:

```python
TransactionID
```

---

# 📥 Dataset Download

The datasets used in this project are publicly available from the IEEE-CIS Fraud Detection competition on Kaggle.

Download datasets from:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Required files:
- train_transaction.csv
- train_identity.csv

After downloading, place both files inside the `data/` folder:

```bash
data/
├── train_transaction.csv
└── train_identity.csv
```

# ⚙️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Imbalanced-learn

---

# 🧠 Machine Learning Models

The following models were trained and evaluated:

- LightGBM Classifier
- XGBoost Classifier
- Isolation Forest

---

# 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

---

# 🔍 Explainable AI (SHAP)

SHAP was used to:
- Identify important fraud features
- Explain model predictions
- Generate SHAP summary plots
- Create transaction-level waterfall plots

---

# 🚦 Risk Segmentation

Transactions were categorized into:

| Risk Tier | Fraud Probability |
|------------|------------------|
| 🔴 Critical Risk | ≥ 0.75 |
| 🟡 Suspicious | 0.40 – 0.74 |
| 🟢 Clear | < 0.40 |

---

# 📊 Visualizations Included

- SHAP Summary Plot
- Fraud Rate by Hour
- Transaction Amount Distribution
- ROC Curve
- Precision-Recall Curve
- Risk Tier Donut Chart
- Interactive Plotly Scatter Plot

---

# 🖥️ Streamlit Dashboard

The project includes a multi-page interactive Streamlit dashboard with:

## 📍 Overview Page
- Total Transactions
- Fraud Count
- Detection Rate
- Average Fraud Amount

## 📍 Transaction Explorer
- Searchable Transactions
- Risk Score Lookup
- Filterable Table

## 📍 SHAP Explainer
- SHAP Waterfall Plot
- Plain-English Fraud Explanation

---

# 🚀 Run Project Locally

## Step 1 — Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3 — Run Notebook

Open:

```bash
analysis.ipynb
```

## Step 4 — Run Streamlit Dashboard

```bash
cd dashboard
streamlit run app.py
```

---

# 🌐 Streamlit Deployment

Deployed using Streamlit Community Cloud.

## Live Dashboard

```bash
Add your Streamlit deployment URL here
```

---

# 💡 Key Insights

- XGBoost achieved the best fraud detection performance.
- PR-AUC proved more reliable than accuracy for imbalanced fraud datasets.
- SHAP analysis identified transaction amount and behavioral features as major fraud indicators.
- Risk segmentation improved fraud monitoring and operational prioritization.

---

# 📌 Future Improvements

- Real-time fraud streaming pipeline
- Deep learning based fraud models
- API integration for live prediction
- Advanced behavioral analytics
- Cloud deployment optimization

---

# 👨‍💻 Author

**Saransh Malik**

---

# ⭐ Conclusion

This project demonstrates a complete end-to-end fraud detection pipeline combining:
- Machine Learning
- Explainable AI
- Interactive Dashboards
- Risk Analytics
- Business Intelligence

The system provides accurate fraud prediction along with interpretable insights for real-world fraud monitoring applications.
