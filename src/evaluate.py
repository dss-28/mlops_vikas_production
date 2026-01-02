# src/evaluate.py

import os
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.logger import get_logger
import yaml

logger = get_logger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

ARTIFACTS_DIR = config["artifacts"]["model_dir"]

def evaluate_model(model, X_test, y_test, save_path=ARTIFACTS_DIR):
    """
    Evaluate the trained model on test data and save metrics.

    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        save_path: Directory to save metrics

    Returns:
        dict: accuracy, classification report, confusion matrix, metrics_path
    """
    logger.info("Starting model evaluation")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON

    logger.info(f"Accuracy: {acc}")
    logger.info(f"Classification Report: {report}")
    logger.info(f"Confusion Matrix: {conf_matrix}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save metrics to JSON
    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }, f, indent=4)
    
    logger.info(f"Metrics saved at: {metrics_path}")

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": conf_matrix,
        "metrics_path": metrics_path
    }

# -------------------------
# Optional standalone test
# -------------------------
if __name__ == "__main__":
    print("evaluate.py is ready. Run it via pipeline.py after training a model.")
