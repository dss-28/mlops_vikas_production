# pipeline.py
from src.data_ingestion import load_data
from src.train import train_pipeline
from src.inference import run_inference
from sklearn.model_selection import train_test_split
import yaml
from src.logger import get_logger
import mlflow

logger = get_logger(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["raw_path"]
TARGET_COL = config["preprocessing"]["target_column"]
TEST_SIZE = config["preprocessing"]["test_size"]
RANDOM_STATE = config["preprocessing"]["random_state"]

MLFLOW_EXPERIMENT = config["mlflow"]["experiment_name"]

if __name__ == "__main__":
    logger.info("Starting full MLflow + pipeline workflow")

    # -----------------------------
    # 0️⃣ Set MLflow experiment
    # -----------------------------
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # 1️⃣ Load data
    df = load_data(DATA_PATH)

    # 2️⃣ Split data
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # 3️⃣ Train pipeline & register in MLflow
    pipeline = train_pipeline(X_train, y_train, X_test, y_test)

    # 4️⃣ Run inference using MLflow Production model
    predictions = run_inference(X_test)
    logger.info(f"Sample predictions: {predictions[:10]}")
    print("Sample predictions:", predictions[:10])
