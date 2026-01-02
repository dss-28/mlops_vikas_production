from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.model_trainer import train_model

df = ingest_data("data/raw/adult.csv")

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

results = train_model(X_train, X_test, y_train, y_test)

print("Accuracy:", results["accuracy"])
print("Model saved at:", results["model_path"])
print(results["report"])
