import streamlit as st
import pickle
import numpy as np
from streamlit_extras.let_it_rain import rain

st.set_page_config(page_title="Loan Prediction", page_icon=":moneybag:", layout="wide")

st.markdown(
    "<h1 style='text-align: center; margin-top: -30px; margin-bottom: 30px;'>Loan Approval Prediction</h1>",
    unsafe_allow_html=True,
)


def encode_property_area(area):
    if area.lower() == 'urban':
        return np.array([1, 0, 0])
    elif area.lower() == 'semiurban':
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

def encode_credit_history(ch):
    return 1 if ch.lower() == 'yes' else 0

# ------------------inputs--------------------- #
gender = st.radio("Select Gender", ["Male", "Female"])
married = st.radio("Are you married?", ["Yes", "No"])
dependents = st.number_input("Number of dependents", min_value=0)
education = st.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.radio("Are you self-employed?", ["Yes", "No"])
applicant_income = st.number_input('Applicant Income in INR', min_value=0)
coapplicant_income = st.number_input("Co-Applicant Income in INR", min_value=0)
loan_amount = st.number_input("Loan Amount in INR", min_value=0)
loan_amount_term = st.selectbox("Loan Amount Term", [60, 90, 120, 180, 240, 360, 480])
credit_history = st.radio("Credit History(Previously taken loan?)", ['Yes', 'No'])
property_area = st.selectbox("Select Property Area", ["Urban", "Semiurban", "Rural"])
# ---------------------------------------------------- #

print("Value of dependents:", dependents)

# Encoding the categorical values
with open(r'D:\Loan Approval Predictor\notebook\label_encoder.pkl', "rb") as file:
    label_encoder = pickle.load(file)

with open(r'D:\Loan Approval Predictor\notebook\ordinal_encoder.pkl', "rb") as file:
    ordinal_encoder = pickle.load(file)

gender_encoded = label_encoder.transform([[gender]])
married_encoded = label_encoder.transform([[married]])
self_employed_encoded = label_encoder.transform([[self_employed]])
education_encoded = ordinal_encoder.transform([[education]])
credit_history_encoded = label_encoder.transform([[credit_history]])

encoded_property_area = encode_property_area(property_area)

encoded_credit_history = encode_credit_history(credit_history)

# Flatten the arrays
gender_encoded = gender_encoded.flatten()
married_encoded = married_encoded.flatten()
self_employed_encoded = self_employed_encoded.flatten()
education_encoded = education_encoded.flatten()
credit_history_encoded = credit_history_encoded.flatten()

# Scaling the values
with open(r'D:\Loan Approval Predictor\notebook\scaler_encoder.pkl', "rb") as file:
    scaler = pickle.load(file)

encoded_features = [
    *gender_encoded,
    *married_encoded,
    dependents,  # No need to flatten dependents as it's not encoded
    *education_encoded,
    *self_employed_encoded,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    encoded_credit_history,
    *encoded_property_area,
]

scaled_data = scaler.transform([encoded_features])

# Predicting the approval status
with open(r'D:\Loan Approval Predictor\notebook\model.pkl', "rb") as file:
    model = pickle.load(file)

result = model.predict(scaled_data)

if st.button("Submit"):
    if result[0].lower() == 'y':
        st.success("Congratulations! Your loan can be approved.")
    elif result[0].lower() == 'n':
        st.error("Sorry, your loan application may not be approved.")