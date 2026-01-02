# src/train.py
import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocessing import build_preprocessor
from src.logger import get_logger

logger = get_logger(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

ARTIFACT_DIR = config["artifacts"]["model_dir"]
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MLFLOW_EXPERIMENT = config["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]

def train_pipeline(X_train, y_train, X_test, y_test):
    logger.info("Building pipeline")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(**config["model"]["params"]))
    ])

    logger.info("Training pipeline")
    pipeline.fit(X_train, y_train)

    logger.info("Predicting on test set")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")

    # -----------------------------
    # Save locally
    # -----------------------------
    local_path = os.path.join(ARTIFACT_DIR, "pipeline.joblib")
    joblib.dump(pipeline, local_path)
    logger.info(f"Pipeline saved locally at {local_path}")

    # -----------------------------
    # Log metrics and model to MLflow
    # -----------------------------
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Log model to MLflow and register
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="pipeline",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        logger.info(f"Pipeline logged and registered in MLflow as '{REGISTERED_MODEL_NAME}'")

        # Optional: Promote to Production automatically
        client = MlflowClient()
        latest_model = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
        client.transition_model_version_stage(
            name=latest_model.name,
            version=latest_model.version,
            stage="Production"
        )
        logger.info(f"Model version {latest_model.version} promoted to Production")

    return pipeline
