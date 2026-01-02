# test_preprocess_logging.py

from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting preprocessing logging test ✅")

    # Load data
    data_path = "data/raw/adult.csv"  # Update path if needed
    df = ingest_data(data_path)
    logger.info("Data loaded successfully")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Log shapes
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    logger.info("Preprocessing logging test completed ✅")
    print("Check logs folder for detailed run info.")

if __name__ == "__main__":
    main()
