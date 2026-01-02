# test_inference.py
import pandas as pd
import joblib
import os

# Load saved pipeline locally
PIPELINE_PATH = os.path.join("artifacts", "pipeline.joblib")
pipeline = joblib.load(PIPELINE_PATH)
print(f"Local pipeline loaded from {PIPELINE_PATH}")

# Load saved feature columns
COLUMNS_PATH = os.path.join("artifacts", "columns.joblib")
feature_cols = joblib.load(COLUMNS_PATH)
numeric_cols = feature_cols["numeric"]
categorical_cols = feature_cols["categorical"]
all_cols = numeric_cols + categorical_cols
print(f"Loaded feature columns: {all_cols}")

# Sample input matching training columns
sample = pd.DataFrame([{
    "age": 39, "fnlwgt": 77516, "education-num": 13, "capital-gain": 0,
    "hours-per-week": 40, "workclass": "State-gov", "education": "Bachelors",
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "race": "White", "sex": "Male", "native-country": "United-States",
    "relationship": "Not-in-family", "capital-loss": 0
}])

# Keep only columns used in training
sample = sample[all_cols]

# Predict
prediction = pipeline.predict(sample)
print("Prediction:", prediction)
