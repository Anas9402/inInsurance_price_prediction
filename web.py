import streamlit as st
import pandas as pd
import joblib

st.markdown("---")
st.caption("Created by [Anas Athar] • Powered by Streamlit & Random Forest")
# Load the trained model
model = joblib.load('insurance.pkl')

# App title
st.title("💰 Insurance Charges Predictor")
st.write("Enter your details to estimate insurance charges:")

# Input fields
age = st.slider("Age", 5, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
height = st.number_input("Height (in cm)", min_value=36.0, max_value=200.0, value=65.0)
weight = st.number_input("Weight (in Kg)", min_value=5.0, max_value=200.0, value=60.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Calculate BMI
height_c=height/100
bmi = weight / (height_c**2)
st.write(f"🧮 Calculated BMI: **{bmi:.2f}**")

# Predict button
if st.button("Predict Charges"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    i=0
    prediction = model.predict(input_data)[i]
    st.success(f"Estimated Insurance Charges: ₹{prediction:,.2f}")