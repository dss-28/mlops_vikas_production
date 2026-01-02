# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import joblib
import os
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

ARTIFACT_DIR = config["artifacts"]["model_dir"]
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def build_preprocessor():
    """
    Returns a ColumnTransformer for numeric and categorical features.
    """
    # Load sample to detect types
    sample_path = config["data"]["raw_path"]
    df = pd.read_csv(sample_path, skipinitialspace=True)
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    
    if config["preprocessing"]["target_column"] in numeric_features:
        numeric_features.remove(config["preprocessing"]["target_column"])
    if config["preprocessing"]["target_column"] in categorical_features:
        categorical_features.remove(config["preprocessing"]["target_column"])

    # Save columns for inference
    columns_path = os.path.join(ARTIFACT_DIR, "columns.joblib")
    joblib.dump({"numeric": numeric_features, "categorical": categorical_features}, columns_path)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
