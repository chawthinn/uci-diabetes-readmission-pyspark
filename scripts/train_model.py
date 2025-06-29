"""
This script trains a readmission prediction model for diabetic patients.
It loads a preprocessed, one-hot–encoded dataset (originally prepared using a PySpark pipeline),
then trains a Random Forest classifier using scikit-learn.

The model and its corresponding feature list are saved as `.pkl` and `.json` files, respectively,
to support deployment in a Streamlit web application. Scikit-learn was chosen for its ease of
integration with Python-based app frameworks, allowing efficient inference without requiring
a Spark environment.

Outputs:
- rf_sklearn_model.pkl : Trained model for prediction
- feature_names.json   : Ordered list of feature names to ensure input alignment
"""

import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Define paths
DATA_PATH = "dataset/diabetes_cleaned_onehot.csv"
MODEL_PATH = "app/rf_sklearn_model.pkl"
FEATURES_PATH = "app/feature_names.json"


# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["readmitted_label"])
y = df["readmitted_label"]

# Save feature names to match during prediction
feature_names = X.columns.tolist()
with open(FEATURES_PATH, "w") as f:
    json.dump(feature_names, f)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Optionally evaluate
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

