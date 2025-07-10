
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# add side bar
st.sidebar.title("Diabetes Readmission App")
st.sidebar.markdown("""
This app predicts **diabetes patient readmission risk** into:  
- **Not Readmitted**  
- **Readmitted After 30 Days**  
- **Readmitted Within 30 Days**  

**Dataset**: [Diabetes 130-US hospitals (UCI)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
""")

# Extract and load the model
import os
APP_DIR = os.path.dirname(__file__)
ZIP_PATH = os.path.join(APP_DIR, "rf_sklearn_model.zip")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(os.path.join(APP_DIR, "model_dir"))

model = joblib.load(os.path.join(APP_DIR, "model_dir/rf_sklearn_model.pkl"))

# Load model and feature info
with open(os.path.join(APP_DIR, "feature_names.json")) as f:
    all_features = json.load(f)

with open(os.path.join(APP_DIR, "top_12_features.json")) as f:
    top_12_features = json.load(f)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section:", [
    "Predict Readmission",
    "Model Evaluation"
])

# Set Constants for dropdown options
DISCHARGE_OPTIONS = {
    1: "Discharged to home or self-care (routine discharge)",
    2: "Discharged/transferred to a short-term general hospital for inpatient care",
    3: "Discharged/transferred to a skilled nursing facility (SNF)",
    4: "Discharged/transferred to an intermediate care facility (ICF)",
    5: "Discharged/transferred to another type of health care institution (e.g., rehab, long-term care)",
    6: "Discharged/transferred to home under care of organized home health service organization",
    7: "Left against medical advice or discontinued care",
    8: "Discharged/transferred to home under care of home IV provider",
    9: "Admitted as an inpatient to another hospital",
    10: "Discharged/transferred to a psychiatric hospital or psychiatric distinct part unit of a hospital",
    12: "Discharged/transferred to Court/Law enforcement (e.g., prison, jail)",
    15: "Discharged/transferred to federal health care facility (e.g., VA hospital)",
    16: "Discharged/transferred to a resident or Assisted Living facility",
    17: "Discharged/transferred to a nursing facility certified under Medicaid but not Medicare",
    22: "Discharged/transferred to a short-term hospital inpatient for therapeutic or palliative reasons (less common)",
    23: "Discharged/transferred to another unspecified medical facility",
    24: "Discharged/transferred to a multidrug infusion therapy home care provider",
    27: "Discharged/transferred to a hospice — home",
}

ADMISSION_SOURCE_OPTIONS = {
    1: "Non-health care facility (physician referral—home/work)",
    2: "Clinic or physician’s office",
    3: "Reserved for national assignment / HMO referral",
    4: "Transfer from another hospital (different facility)",
    5: "Transfer from a skilled nursing/intermediate care facility (SNF/ICF)",
    6: "Transfer from another type of health care facility",
    7: "Emergency Room admission (code discontinued by CMS in 2010, but still present historically)",
    8: "Court or law enforcement (jail/prison admission)",
    9: "Not available / unknown/invalid",
    10: "Newborn – point of origin = type of delivery",
    11: "HMO referral (sometimes used interchangeably with code 3, depending on CMS updates)",
    13: "Transfer from hospital-based hospice (rare)",
    14: "Transfer from clinic-based hospice (rare)",
    17: "Transfer from ambulatory surgery center or birthing center (some CMS sources include)",
    20: "Transfer from acute psychiatric hospital/unit (distinct CMS category)",
    22: "Transfer from intermediate psychiatric facility",
    25: "Transfer from another long-term care hospital or facility (e.g., inpatient rehab)",
}


if page == "Predict Readmission":
    st.title("Diabetes Readmission Risk Prediction")
    st.markdown("This tool predicts the **readmission risk** of a diabetes patient based on selected inputs.")

    st.header("Patient Details")
    user_input = {}
    for feature in top_12_features:
        if feature == "discharge_disposition_id":
            discharge_labels = [f"{id} - {desc}" for id, desc in DISCHARGE_OPTIONS.items()]
            selected_label = st.selectbox("Discharge Disposition ID", discharge_labels)
            user_input[feature] = int(selected_label.split(" - ")[0])
        elif feature == "admission_source_id":
            admission_source_labels = [f"{id} - {desc}" for id, desc in ADMISSION_SOURCE_OPTIONS.items()]
            selected_label = st.selectbox("Admission Source ID", admission_source_labels)
            user_input[feature] = int(selected_label.split(" - ")[0])
        elif "num" in feature or "number" in feature or "time" in feature:
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

    input_dict = {feature: 0 for feature in all_features}
    for key in user_input:
        input_dict[key] = user_input[key]
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Readmission Risk"):
        prediction = model.predict(input_df)[0]
        label_map = {
            0: "Readmitted After 30 Days",
            1: "Readmitted Within 30 Days",
            2: "Not Readmitted"
        }
        st.subheader("Prediction Result:")
        st.success(f"🧾 The model predicts: **{label_map[prediction]}**")

elif page == "Model Evaluation":
    st.title("Model Evaluation on Test Set")
    try:
        test_df = pd.read_csv(os.path.join(APP_DIR, "diabetes_cleaned_onehot.csv"))
        X_test = test_df.drop(columns=[col for col in test_df.columns if col.startswith("readmitted")])
        y_test = test_df["readmitted_label"]
        y_pred = model.predict(X_test)

        st.subheader("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=False)
        st.code(report)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['NO', '>30', '<30'], yticklabels=['NO', '>30', '<30'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display evaluation metrics: {e}")