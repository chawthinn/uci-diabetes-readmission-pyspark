import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# 1. Load your cleaned, one-hot encoded dataset
df = pd.read_csv("diabetes_cleaned_onehot.csv")

# 2. Split features and target
X = df.drop(columns=["readmitted_label"])
y = df["readmitted_label"]

# 3. Save feature names for use in Streamlit
with open("feature_names.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define pipeline: RandomOverSampler + RandomForest
pipeline = Pipeline(steps=[
    ("oversample", RandomOverSampler(sampling_strategy="not majority", random_state=42)),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 6. Fit model (resampling happens inside)
pipeline.fit(X_train, y_train)

# 7. Evaluate on original (unmodified) test set
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save the trained pipeline
joblib.dump(pipeline, "rf_sklearn_model.pkl")
