import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and feature info
model = joblib.load("rf_sklearn_model.pkl")

with open("feature_names.json") as f:
    all_features = json.load(f)

with open("top_12_features.json") as f:
    top_12_features = json.load(f)

# UI Header
st.title("🩺 Diabetes Readmission Risk Prediction")
st.markdown("This tool predicts the **readmission risk** of a diabetes patient based on selected inputs.")

# Collect input for top 12 features
st.header("Patient Details")
user_input = {}

for feature in top_12_features:
    if "num" in feature or "number" in feature or "time" in feature:
        user_input[feature] = st.slider(f"{feature}", 0, 50, 1)
    elif feature.startswith("gender_"):
        user_input[feature] = st.radio("Gender", ["Male", "Female"]) == feature.split("_")[1]
    elif feature.startswith("race_"):
        race_options = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
        selected = st.selectbox("Race", race_options)
        user_input[feature] = int(selected == feature.split("_")[1])
    elif feature.startswith("insulin_"):
        selected = st.selectbox("Insulin Level", ["No", "Up", "Down", "Steady"])
        user_input[feature] = int(selected == feature.split("_")[1])
    elif feature.startswith("metformin_"):
        selected = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
        user_input[feature] = int(selected == feature.split("_")[1])
    elif feature.startswith("A1Cresult_"):
        selected = st.selectbox("A1C Result", [">7", ">8", "Norm", "Unknown"])
        user_input[feature] = int(selected == feature.split("_")[1])
    else:
        user_input[feature] = st.number_input(f"{feature}", 0.0, 100.0, step=1.0)

# Build input row with ALL expected features (initialize to 0)
input_dict = {feature: 0 for feature in all_features}
for key in user_input:
    input_dict[key] = user_input[key]

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Readmission Risk"):
    prediction = model.predict(input_df)[0]
    label_map = {
        0: "Readmitted After 30 Days",
        1: "Readmitted Within 30 Days",
        2: "Not Readmitted"
    }
    st.subheader("Prediction Result:")
    st.success(f"🧾 The model predicts: **{label_map[prediction]}**")
