import streamlit as st
from pyspark.ml import PipelineModel
from utils.feature_builder import build_feature_vector
from utils.spark_session import get_spark_session

# Initialize Spark
spark = get_spark_session()

# Load model
model = PipelineModel.load("rf_pipeline_model")

st.title("🩺 Diabetes Readmission Prediction (Local PySpark)")
st.write("Enter patient info to predict 30-day readmission risk.")

# --- UI Form Inputs
age_num = st.slider("Age (Numeric)", 20, 90, 60)
discharge_id = st.selectbox("Discharge Disposition ID", [1, 2, 3, 4, 5, 6, 7])
admission_id = st.selectbox("Admission Type ID", [1, 2, 3, 4, 5, 6, 7])
source_id = st.selectbox("Admission Source ID", [1, 2, 3, 4, 5, 6, 7])
time_in_hospital = st.slider("Time in Hospital (days)", 1, 20, 3)
num_medications = st.number_input("Number of Medications", 0, 100, 10)
num_lab_procedures = st.number_input("Lab Procedures", 0, 100, 45)
num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 9)
num_inpatient = st.slider("Inpatient Visits (past year)", 0, 20, 0)
num_emergency = st.slider("Emergency Visits (past year)", 0, 20, 0)
num_outpatient = st.slider("Outpatient Visits (past year)", 0, 20, 0)
insulin = st.selectbox("Insulin Status", ["No", "Steady", "Up", "Down"])
metformin = st.selectbox("Metformin Status", ["No", "Steady", "Up", "Down"])

# --- Predict Button
if st.button("Predict"):
    input_dict = {
        "age_num": age_num,
        "discharge_disposition_id": discharge_id,
        "admission_type_id": admission_id,
        "admission_source_id": source_id,
        "time_in_hospital": time_in_hospital,
        "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures,
        "number_diagnoses": num_diagnoses,
        "number_inpatient": num_inpatient,
        "number_emergency": num_emergency,
        "number_outpatient": num_outpatient,
        "insulin": insulin,
        "metformin": metformin
    }

    feature_df = build_feature_vector(input_dict, spark)
    prediction = model.transform(feature_df).select("prediction").collect()[0]["prediction"]
    label = "🔴 Readmitted within 30 Days" if prediction == 1 else "🟢 Not Readmitted"
    st.subheader("Prediction Result:")
    st.success(label)
