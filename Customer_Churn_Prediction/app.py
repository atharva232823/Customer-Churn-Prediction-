import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load preprocessing + model
# -----------------------------
scaler = pickle.load(open("scaler.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))
ordinal_contract = pickle.load(open("ordinal_encoder_contract.pkl", "rb"))
ordinal_internet = pickle.load(open("ordinal_encoder_internet.pkl", "rb"))
ohe = pickle.load(open("ohe_encoder.pkl", "rb"))
model = pickle.load(open("svc_model.pkl", "rb"))

# -----------------------------
# Load dataset
# -----------------------------
df_data = pd.read_csv(r"C:\Users\Ūśēr̥\Downloads\Telco-Customer-Churn.csv")
df_data["TotalCharges"] = pd.to_numeric(df_data["TotalCharges"], errors="coerce")

# -----------------------------
# GLOBAL STYLES (Modern UI)
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: -10px;
        }
        .sub-title {
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 25px;
        }
        .card {
            background-color: #ffffff;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            border: 1px solid #f0f0f0;
            margin-bottom: 14px;
        }
        .card-header {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 8px;
        }
        .small-label label {
            font-size: 0.86rem !important;
        }
        div[class*="stSelectbox"], div[class*="stSlider"] {
            margin-bottom: 6px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether a customer is likely to churn using your trained SVC model.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# LAYOUT: 3 COLUMNS
# ============================================================
left_col, mid_col, right_col = st.columns([1.3, 1.8, 1.2])

# ============================================================
# LEFT COLUMN — CUSTOMER INFO
# ============================================================
with left_col:
    st.markdown('<div class="card small-label">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">👤 Customer Details</div>', unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_choice = st.selectbox("Senior Citizen", ["No", "Yes"])
    SeniorCitizen = 1 if senior_choice == "Yes" else 0
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Billing info card
    st.markdown('<div class="card small-label">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">💰 Billing Information</div>', unsafe_allow_html=True)

    # ----------------------------
    # UPDATED SLIDERS WITH STEP
    # ----------------------------
    c1, c2 = st.columns(2)

    with c1:
        tenure = st.slider("Tenure (Months)", 0, 100, 12)
        MonthlyCharges = st.slider("Monthly Charges", 0, 120, 50)

    with c2:
        TotalCharges = st.slider("Total Charges", 0, 10000, 1000, step=50)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MIDDLE COLUMN — SERVICES + CONTRACT
# ============================================================
with mid_col:
    st.markdown('<div class="card small-label">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📡 Services Subscribed</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    with s2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    with s3:
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Contract card
    st.markdown('<div class="card small-label">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">💳 Contract & Payment</div>', unsafe_allow_html=True)

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================
input_df = pd.DataFrame({
    "gender": [gender],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "PhoneService": [PhoneService],
    "PaperlessBilling": [PaperlessBilling],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "PaymentMethod": [PaymentMethod],
    "MultipleLines": [MultipleLines],
    "Contract": [Contract],
    "InternetService": [InternetService],
    "SeniorCitizen": [SeniorCitizen],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
})

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
contract_encoded = ordinal_contract.transform(input_df[["Contract"]])
internet_encoded = ordinal_internet.transform(input_df[["InternetService"]])

one_hot_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaymentMethod", "MultipleLines",
]
ohe_encoded = ohe.transform(input_df[one_hot_cols])

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
num_imputed = imputer.transform(input_df[numeric_cols])

final_before_scaling = np.hstack([
    contract_encoded,
    internet_encoded,
    ohe_encoded,
    num_imputed,
])

final_scaled = scaler.transform(final_before_scaling)

# ============================================================
# RIGHT COLUMN — IMAGE + PREDICTION
# ============================================================
with right_col:

    st.image("banner.png", use_container_width=True)

    st.markdown('<div class="card small-label">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🔮 Prediction</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Customer Churn", use_container_width=True)

    if predict_btn:
        prediction = model.predict(final_scaled)[0]

        try:
            decision_score = model.decision_function(final_scaled)[0]
            churn_risk = 1 / (1 + np.exp(-decision_score))
            churn_risk_pct = int(round(churn_risk * 100))
        except:
            churn_risk_pct = 50

        if prediction == "Yes":
            st.error("🔴 Customer is LIKELY to Churn")
        else:
            st.success("🟢 Customer is NOT Likely to Churn")

        st.markdown("---")
        st.markdown("### 📈 Churn Risk Gauge")
        st.progress(churn_risk_pct)

        a, b = st.columns(2)
        with a:
            st.metric("Estimated Risk", f"{churn_risk_pct}%")
        with b:
            st.metric("Prediction", "Churn" if prediction == "Yes" else "No Churn")

    else:
        st.info("Click the button to generate a prediction.")

    st.markdown("</div>", unsafe_allow_html=True)
