import pandas as pd
import json

def build_feature_vector(user_inputs: dict, feature_names_path: str = "feature_names.json"
) -> pd.DataFrame:
    """
    Builds a model-ready feature vector (107 columns) using minimal user input,
    inferred defaults, and one-hot encoding.
    """
    # Load the expected feature names (from training)
    with open(feature_names_path, "r") as f:
        expected_columns = json.load(f)

    # Sensible defaults for missing fields
    defaults = {
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "insulin": "No",
        "change": "No",
        "diabetesMed": "Yes"
    }

    # Merge defaults with user input (user overrides defaults)
    full_input = {**defaults, **user_inputs}

    # Handle derived features
    if "age" in full_input:
        age_map = {
            "[0-10]": 5, "[10-20]": 15, "[20-30]": 25, "[30-40]": 35,
            "[40-50]": 45, "[50-60]": 55, "[60-70]": 65, "[70-80]": 75,
            "[80-90]": 85, "[90-100]": 95
        }
        full_input["age_num"] = age_map.get(full_input["age"], 50)

    # Create DataFrame
    df = pd.DataFrame([full_input])

    # One-hot encode all applicable fields
    df_encoded = pd.get_dummies(df)

    # Reindex to full 107 columns and fill missing with 0
    df_final = df_encoded.reindex(columns=expected_columns, fill_value=0)

    return df_final
