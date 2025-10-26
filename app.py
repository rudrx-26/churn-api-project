import streamlit as st
import random

st.title("Customer Churn Prediction (Demo)")

with st.form("user_input_form"):
    credit_score = st.number_input("Credit Score", value=650)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", value=40)
    tenure = st.number_input("Tenure", value=5)
    balance = st.number_input("Balance", value=50000)
    num_products = st.number_input("Num Of Products", value=2)
    has_card = st.selectbox("Has Credit Card (1=Yes, 0=No)", [1, 0])
    is_active = st.selectbox("Is Active Member (1=Yes, 0=No)", [1, 0])
    salary = st.number_input("Estimated Salary", value=90000)
    geo_germany = st.selectbox("Geography: Germany (1=Yes, 0=No)", [0, 1])
    geo_spain = st.selectbox("Geography: Spain (1=Yes, 0=No)", [0, 1])
    submitted = st.form_submit_button("Predict")

if submitted:
    prob = random.uniform(0, 1)
    pred = int(prob > 0.5)
    st.success(f"Churn prediction: **{'YES' if pred else 'NO'}**  (Probability: {prob:.2f})")
    st.caption("This is a demo prototype—when deployed on a bigger server, your saved ML model predicts here!")

