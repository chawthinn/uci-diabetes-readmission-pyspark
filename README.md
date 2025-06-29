# uci-diabetes-readmission-pyspark	
This repository is for a group project as part of the Big Data Framework course. It focuses on building a predictive machine learning model using PySpark on the UCI "Diabetes 130-US hospitals for years 1999–2008" dataset. The goal is to predict patient readmission by performing data preprocessing, transformation, feature engineering, and model training using scalable Spark ML techniques.

# Folder Structure 

```
uci-diabetes-readmission-pyspark/
├── app/                        ← Streamlit + PySpark app lives here
│   ├── app.py                  ← Main Streamlit UI
│   ├── rf_pipeline_model/      ← PySpark pipeline model
│   └── utils/
│       ├── spark_session.py
│       └── feature_builder.py
│
├── datasets/                  ←  Dataset
├── scripts/                   ←  EDA, training, notebook merge logic
├── merge_notebooks.py
├── README.md
├── LICENSE
├── requirements.txt
```
