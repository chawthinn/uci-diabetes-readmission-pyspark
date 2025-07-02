# uci-diabetes-readmission-pyspark	
This repository is for a group project as part of the Big Data Framework course. It focuses on building a predictive machine learning model using PySpark on the UCI "Diabetes 130-US hospitals for years 1999–2008" dataset. The goal is to predict patient readmission by performing data preprocessing, transformation, feature engineering, and model training using scalable Spark ML techniques.

# Folder Structure 

```
uci-diabetes-readmission-pyspark/
├── app/                                         ← Streamlit + PySpark app lives here
│   ├── app.py                                   ← Main Streamlit UI
│   ├── rf_pipeline_model/                       ← PySpark pipeline model
│   ├── train_model.py                           ← Script to train and export model
│   └── utils/
│       └── preprocess.py                        # build_feature_vector() → saves feature_names.json
│
├── datasets/                                    ← dataset
│
├── scripts/                                     ← EDA, training experiments, merging notebooks
│
├── final_submission/                            ← ✅ Final deliverables (ready to submit)
│   ├── diabetic_data.csv                        ← Raw Dataset
|   ├── one hot encoding                         ← One Hot Encoding File
│   ├── DiabetesReadmissionPPT.pptx              ← Presentation slides
|   ├── Diabetes Readmission Report.docx         ← Final Report Document
│   ├── part1_preprocessing_encoding.ipynb       ← Final cleaning, EDA and preprocessing code
│   └── part2_model_training_evaluations.ipynb   ← Final modeling and training evaluations code
│
├── merge_notebooks.py
├── README.md
├── LICENSE
├── requirements.txt
```

Note :

We initially developed the machine learning pipeline using PySpark to handle large-scale preprocessing and feature engineering efficiently. PySpark's DataFrame-based transformations were well-suited for cleaning, encoding, and preparing the dataset in a distributed manner.

However, for deployment, especially in a Streamlit-based web application, we retrained the model using scikit-learn. Scikit-learn makes it easy to serialize models into .pkl files, which are lightweight and can be directly loaded in a Python web environment. This made the model portable and deployable without needing a Spark cluster.

The trained model file (`rf_sklearn_model.pkl`) is approximately 400 MB in size and was not uploaded to the repository due to file size limitations. To generate the model locally,
please run this script after placing the cleaned dataset in the correct path.
