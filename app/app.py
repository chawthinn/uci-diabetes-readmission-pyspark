import streamlit as st
import joblib
import numpy as np
from utils.preprocess import build_feature_vector
import json


# Load expected column order
with open("feature_names.json", "r") as f:
    expected_columns = json.load(f)

# Load trained Scikit-learn model
model = joblib.load("rf_sklearn_model.pkl")

st.title("🩺 Diabetes Readmission Risk Predictor")
st.write("Predict whether a diabetic patient will be readmitted within 30 days after discharge.")

# Top 10 user inputs (based on feature importance)
number_inpatient = st.number_input("Number of inpatient visits", min_value=0)
gender = st.selectbox("Gender", ["Male", "Female"])
race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
number_diagnoses = st.number_input("Number of diagnoses", min_value=0)
num_medications = st.number_input("Number of medications", min_value=0)
time_in_hospital = st.number_input("Time in hospital (days)", min_value=1)
num_lab_procedures = st.number_input("Number of lab procedures", min_value=0)
number_emergency = st.number_input("Number of emergency visits", min_value=0)
number_outpatient = st.number_input("Number of outpatient visits", min_value=0)
num_procedures = st.number_input("Number of procedures", min_value=0)
age = st.selectbox("Age group", ["[0-10]", "[10-20]", "[20-30]", "[30-40]", "[40-50]", "[50-60]", "[60-70]", "[70-80]", "[80-90]", "[90-100]"])
A1Cresult = st.selectbox("A1C Result", [">7", ">8", "Norm", "None"])
max_glu_serum = st.selectbox("Max Glucose Serum", [">200", ">300", "Norm", "None"])
diabetesMed = st.selectbox("Diabetes Medication Prescribed", ["Yes", "No"])

# Create input dictionary
input_dict = {
    "number_inpatient": number_inpatient,
    "gender": gender,
    "race": race,
    "number_diagnoses": number_diagnoses,
    "num_medications": num_medications,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "number_emergency": number_emergency,
    "number_outpatient": number_outpatient,
    "num_procedures": num_procedures,
    "admission_type_id": 1,  # Default or placeholder if missing from UI
    "age": age,
    "insulin": insulin,
    "metformin": metformin,
    "A1Cresult": A1Cresult,
    "max_glu_serum": max_glu_serum,
    "diabetesMed": diabetesMed
}

if st.button("Predict Readmission"):
    # Build feature vector
    full_vector = build_feature_vector(input_dict)

    # Reindex to match training features
    full_vector = full_vector.reindex(columns=expected_columns, fill_value=0)

    # Predict
    prediction = model.predict(full_vector)[0]

    if prediction == "<30":
        st.error("⚠️ High Risk: Patient likely to be readmitted within 30 days.")
    elif prediction == ">30":
        st.warning("⚠️ Medium Risk: Patient likely to be readmitted after 30 days.")
    else:
        st.success("✅ Low Risk: No readmission expected.")


