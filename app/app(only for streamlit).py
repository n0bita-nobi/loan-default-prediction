import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Loan Default Prediction System")
st.caption("ANN based decision support system for loan risk assessment")

# ------------------ PATH SETUP (CRITICAL FIX) ------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------ LOAD MODEL & ARTIFACTS ------------------

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(BASE_DIR / "model" / "model.h5")

    scaler = pickle.load(open(BASE_DIR / "pickle" / "scaler.pkl", "rb"))
    edu_enc = pickle.load(open(BASE_DIR / "pickle" / "education_encoder.pkl", "rb"))
    mort_enc = pickle.load(open(BASE_DIR / "pickle" / "HasMortgage_encoder.pkl", "rb"))
    dep_enc = pickle.load(open(BASE_DIR / "pickle" / "HasDependents_encoder.pkl", "rb"))
    cos_enc = pickle.load(open(BASE_DIR / "pickle" / "HasCoSigner_encoder.pkl", "rb"))
    emp_enc = pickle.load(open(BASE_DIR / "pickle" / "employment_encoder.pkl", "rb"))
    mar_enc = pickle.load(open(BASE_DIR / "pickle" / "marital_encoder.pkl", "rb"))
    loan_enc = pickle.load(open(BASE_DIR / "pickle" / "loanpurpose_encoder.pkl", "rb"))

    return model, scaler, edu_enc, mort_enc, dep_enc, cos_enc, emp_enc, mar_enc, loan_enc


model, scaler, edu_enc, mort_enc, dep_enc, cos_enc, emp_enc, mar_enc, loan_enc = load_artifacts()

# ------------------ USER INPUT ------------------

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 75)
    income = st.number_input("Annual Income", 0)
    loan_amount = st.number_input("Loan Amount", 0)
    credit_score = st.number_input("Credit Score", 300, 850)
    months_employed = st.number_input("Months Employed", 0)

with col2:
    num_credit_lines = st.number_input("Number of Credit Lines", 0)
    interest_rate = st.number_input("Interest Rate (%)", 0.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti = st.slider("Debt to Income Ratio", 0.0, 1.0)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

employment = st.selectbox("Employment Type", emp_enc.classes_)
marital = st.selectbox("Marital Status", mar_enc.classes_)
loan_purpose = st.selectbox("Loan Purpose", loan_enc.classes_)

has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
has_cosigner = st.selectbox("Has Co-signer", ["Yes", "No"])

# ------------------ PREDICTION ------------------

if st.button("Predict Loan Risk"):
    # Encode categorical features
    edu_val = edu_enc.transform([[education]])[0][0]
    mort_val = mort_enc.transform([has_mortgage])[0]
    dep_val = dep_enc.transform([has_dependents])[0]
    cos_val = cos_enc.transform([has_cosigner])[0]

    emp_id = emp_enc.transform([employment])[0]
    mar_id = mar_enc.transform([marital])[0]
    loan_id = loan_enc.transform([loan_purpose])[0]

    # Scale numerical features (MUST match training)
    numeric = scaler.transform([[
        age,
        income,
        loan_amount,
        credit_score,
        months_employed,
        num_credit_lines,
        interest_rate,
        loan_term,
        dti
    ]])

    prediction = model.predict([
        numeric[:, 0],   # Age
        numeric[:, 1],   # Income
        numeric[:, 2],   # LoanAmount
        numeric[:, 3],   # CreditScore
        numeric[:, 4],   # MonthsEmployed
        numeric[:, 5],   # NumCreditLines
        numeric[:, 6],   # InterestRate
        numeric[:, 7],   # LoanTerm
        numeric[:, 8],   # DTIRatio
        np.array([edu_val]),
        np.array([mort_val]),
        np.array([dep_val]),
        np.array([cos_val]),
        np.array([emp_id]),
        np.array([mar_id]),
        np.array([loan_id])
    ])

    prob = float(prediction[0][0])

    st.subheader("Prediction Result")

    if prob >= 0.5:
        st.error(f"⚠️ High Risk of Default\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Default\n\nProbability: {prob:.2f}")

    st.caption("Prediction is probabilistic and should be used as decision support.")
