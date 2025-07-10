# UCI Diabetes Readmission Multiclass Prediction

This repository is for a group project as part of the Big Data Framework course. It focuses on building a predictive machine learning model using PySpark on the UCI "Diabetes 130-US hospitals for years 1999–2008" dataset. The goal is to predict patient readmission by performing data preprocessing, transformation, feature engineering, and model training using scalable Spark ML techniques.

---

## How to Run the Streamlit App  

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app from the `app` directory:
   ```bash
   cd app
   streamlit run app.py
   ```
3. The app will open in your browser at `http://localhost:8501`.  

The app provides:  
- **Prediction Tool**: Predict diabetes patient readmission risk.  
- **Model Evaluation**: View metrics and confusion matrix for the trained model.  

---

# Folder Structure 
```
uci-diabetes-readmission-pyspark/
├── app/                                         ← Streamlit app (run `streamlit run app.py` here)
│   ├── app.py                                   ← Main Streamlit UI
│   ├── rf_sklearn_model.zip                     ← Zipped scikit-learn model used for predictions
│   ├── diabetes_cleaned_onehot.csv              ← Cleaned dataset for model evaluation
│   ├── feature_names.json                       ← Full feature names used by the model
│   ├── top_12_features.json                     ← Top features for simplified prediction inputs
│   ├── train_model.py                           ← Script to retrain and export the model if needed
│   ├── requirements.txt                         ← Fixed dependencies for deployment
│   ├── runtime.txt                              ← Pins Python version to 3.10 for compatibility
│
├── datasets/                                    ← Raw and processed datasets
│
├── scripts/                                     ← EDA, training experiments
│
├── final_submission/                            ← ✅ FINAL DELIVERABLES (FOR SUBMISSION)
│   ├── diabetic_data.csv                        ← Raw Dataset
│   ├── one hot encoding                         ← One Hot Encoding File
│   ├── DiabetesReadmissionPPT.pptx              ← Presentation slides
│   ├── Diabetes Readmission Report.docx         ← Final Report Document
│   ├── part1_preprocessing_encoding.ipynb       ← Detailed EDA and preprocessing (Part 1)
│   └── part2_model_training_evaluations.ipynb   ← Model training and evaluation (Part 2)
│
├── README.md
├── LICENSE
```

---

**Note**:  
We initially developed the machine learning pipeline using PySpark to handle large-scale preprocessing and feature engineering efficiently. PySpark's DataFrame-based transformations were well-suited for cleaning, encoding, and preparing the dataset in a distributed manner.

However, for deployment, especially in a Streamlit-based web application, we retrained the model using scikit-learn. Scikit-learn makes it easy to serialize models into .pkl files, which are lightweight and can be directly loaded in a Python web environment. This made the model portable and deployable without needing a Spark cluster.

The trained model file (`rf_sklearn_model.pkl`) is approximately 400 MB and could not be uploaded directly to the repository due to size restrictions.

Instead, we have included a zipped version (`rf_sklearn_model.zip`) in the repository. The app automatically unzips and loads the model during runtime in app.py.

You can try the live Streamlit app here: https://uci-diabetes-readmission-pyspark-multiclass.streamlit.app/