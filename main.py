# python -m streamlit run main.py (for run streamlit app in browser)
# pip install scikit-learn
# pip install streamlit
# pip install joblib


import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model (2).pkl")
scaler = joblib.load("scaler.pkl")

x_gender = joblib.load("x_gender (1).pkl")
x_married = joblib.load("x_married (1).pkl")
x_dependents = joblib.load("x_dependents (1).pkl")
x_education = joblib.load("x_education (1).pkl")
x_self_employed = joblib.load("x_self_employed (1).pkl")
x_property_area = joblib.load("x_property_area (1).pkl")
y_encoded = joblib.load("y_encoded (1).pkl")

st.title("Loan Approval Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

if st.button("Predict"):

    gender = x_gender.transform([[gender]])[0][0]
    married = x_married.transform([[married]])[0][0]
    dependents = x_dependents.transform([[dependents]])[0][0]
    education = x_education.transform([[education]])[0][0]
    self_employed = x_self_employed.transform([[self_employed]])[0][0]
    property_area = x_property_area.transform([[property_area]])[0][0]

    input_data = pd.DataFrame([[gender,married,dependents,education,self_employed,applicant_income,
        coapplicant_income,loan_amount,loan_amount_term, credit_history,property_area]], 
        columns=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome",
        "CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    result = y_encoded.inverse_transform(prediction)[0]

    if result == "Y":
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")