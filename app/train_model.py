import pandas as pd
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# 1. Load your dataset
df = pd.read_csv("diabetes_cleaned_onehot.csv")

# 2. Split into features and label
X = df.drop(columns=["readmitted_label"])
y = df["readmitted_label"]

# 3. Save feature names
with open("feature_names.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Define pipeline
pipeline = Pipeline(steps=[
    ("feature_select", SelectKBest(score_func=f_classif, k=30)),
    ("oversample", RandomOverSampler(sampling_strategy={2: 8000}, random_state=42)),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 6. Define parameter grid for Random Forest
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3]
}

# 7. Run GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 8. Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))

# 9. Save final model
joblib.dump(best_model, "rf_sklearn_model.pkl")

# Get feature selector and model from pipeline
selector = best_model.named_steps["feature_select"]
rf_model = best_model.named_steps["model"]

# Get selected feature names
selected_indices = selector.get_support(indices=True)
original_feature_names = X.columns
selected_feature_names = original_feature_names[selected_indices]

# Match importances to selected features
importances = rf_model.feature_importances_
top_17_indices = importances.argsort()[::-1][:17]

# Final: Top 12 feature names (excluding 'age_num')
top_12_features = [
    selected_feature_names[i]
    for i in top_17_indices
    if selected_feature_names[i] != "age_num"
][:12]


# Save to file for UI use
with open("top_12_features.json", "w") as f:
    json.dump(top_12_features, f)


pipeline = Pipeline(steps=[
    ("drop_constant", VarianceThreshold(threshold=0.0)),  # 🔹 removes constant columns
    ("feature_select", SelectKBest(score_func=f_classif, k=30)),
    ("oversample", RandomOverSampler(sampling_strategy={2: 8000}, random_state=42)),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])