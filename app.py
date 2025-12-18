import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("best_churn_model.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")

st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")

# ======================
# USER INPUTS
# ======================

gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = 1 if st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0

Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = int(st.number_input("Tenure (bulan)", min_value=0, value=1))

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = float(st.number_input("Monthly Charges", min_value=0.0, value=50.0))

# 🔴 IMPORTANT: STRING, NOT FLOAT
TotalCharges = st.text_input("Total Charges", value="50.0")

# ======================
# CREATE DATAFRAME (ORDER + TYPES MATCH MODEL)
# ======================
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges  # STRING
}])

# ======================
# PREDICTION
# ======================
if st.button("🔮 Predict Churn"):
    proba_churn = model.predict_proba(input_df)[0][1]

    THRESHOLD = 0.35  # churn-optimized threshold

    if proba_churn >= THRESHOLD:
        st.error(f"⚠️ Pelanggan diprediksi **CHURN** ({proba_churn:.2%})")
    else:
        st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** ({proba_churn:.2%})")

    with st.expander("🔍 Debug"):
        st.write(input_df)
        st.write(input_df.dtypes)
