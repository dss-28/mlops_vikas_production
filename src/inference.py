# src/inference.py
import pandas as pd
import mlflow.sklearn
from src.logger import get_logger
import os
import yaml
import joblib

logger = get_logger(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

ARTIFACT_DIR = config["artifacts"]["model_dir"]
TARGET_COL = config["preprocessing"]["target_column"]

# MLflow registry settings
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]
STAGE = "Production"

# Load feature columns (still local)
COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "columns.joblib")
cols = joblib.load(COLUMNS_PATH)
numeric_features = cols["numeric"]
categorical_features = cols["categorical"]

def run_inference(input_df: pd.DataFrame):
    """
    Run inference using the MLflow-registered Production model.
    """
    # Keep only expected columns
    input_df = input_df[numeric_features + categorical_features]

    # Drop target if exists
    if TARGET_COL in input_df.columns:
        input_df = input_df.drop(TARGET_COL, axis=1)

    logger.info("Loading Production model from MLflow Registry")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{STAGE}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    logger.info("Running predictions")
    predictions = pipeline.predict(input_df)
    logger.info(f"Predictions (first 10): {predictions[:10]}")
    return predictions.tolist()

# Test standalone
if __name__ == "__main__":
    sample = pd.DataFrame([{
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }])
    preds = run_inference(sample)
    print("Predictions:", preds)
