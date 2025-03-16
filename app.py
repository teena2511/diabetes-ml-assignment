import streamlit as st
import pandas as pd
import joblib

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction App")
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=200.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=1000.0)
age = st.number_input("Age", min_value=0, max_value=120)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[glucose, bmi, insulin, age, dpf]], 
                              columns=['Glucose', 'BMI', 'Insulin', 'Age', 'DiabetesPedigreeFunction'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.error(f"Prediction: Positive (Diabetes Risk) with {probability:.2%} probability")
    else:
        st.success(f"Prediction: Negative (No Diabetes Risk) with {1-probability:.2%} probability")