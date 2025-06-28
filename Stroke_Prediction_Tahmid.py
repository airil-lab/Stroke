
import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load("best_rf_model.pkl")  # Ensure this file is in the same directory

# Function to make predictions
def predict(input_data):
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])
    return prediction[0], probability[0][1]

# Streamlit app interface
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Predictor using Random Forest")

st.markdown("""
Welcome to the **Stroke Risk Predictor**.  
Please enter the required health parameters to assess your risk of stroke.
""")

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Encode categorical variables manually (should match training encoding)
gender_dict = {"Male": 1, "Female": 0, "Other": 2}
hypertension_dict = {"No": 0, "Yes": 1}
heart_disease_dict = {"No": 0, "Yes": 1}
ever_married_dict = {"No": 0, "Yes": 1}
work_type_dict = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 1, "Never_worked": 4}
residence_type_dict = {"Urban": 1, "Rural": 0}
smoking_status_dict = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}

# Convert inputs to model-ready format
input_data = [
    gender_dict[gender],
    age,
    hypertension_dict[hypertension],
    heart_disease_dict[heart_disease],
    ever_married_dict[ever_married],
    work_type_dict[work_type],
    residence_type_dict[Residence_type],
    avg_glucose_level,
    bmi,
    smoking_status_dict[smoking_status]
]

# Prediction
if st.button("🔍 Predict Stroke Risk"):
    prediction, probability = predict(input_data)

    if prediction == 1:
        st.error("⚠️ High Risk: You may be at risk of stroke. Please consult a healthcare professional.")
        st.write(f"Model Confidence: **{probability * 100:.2f}%** chance of stroke.")
    else:
        st.success("✅ Low Risk: You are not likely at risk of stroke based on the current data.")
        st.write(f"Model Confidence: **{(1 - probability) * 100:.2f}%** chance of being safe.")

