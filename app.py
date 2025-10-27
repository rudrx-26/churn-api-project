import sentry_sdk
sentry_sdk.init(
    dsn="https://59779dcff8aafa598ede055ada86bcb1@o4510262460481536.ingest.us.sentry.io/4510262500589568",
    traces_sample_rate=1.0
)
1/0


import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.title("Customer Churn Prediction")

# Load model and encoders/scaler (cache for performance)
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_ann_model.h5")
    le_gender = pickle.load(open("le_gender.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, le_gender, scaler

model, le_gender, scaler = load_artifacts()

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
    # Encode categorical variables (adjust according to your training!)
    gender_encoded = le_gender.transform([gender])[0]
    input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                            num_products, has_card, is_active, salary, geo_germany, geo_spain]], dtype=np.float32)
    input_scaled = scaler.transform(input_data)
    prob = float(model.predict(input_scaled)[0][0])
    pred = int(prob > 0.5)
    st.success(f"Churn prediction: **{'YES' if pred else 'NO'}** (Probability: {prob:.2f})")



