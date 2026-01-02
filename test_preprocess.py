from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data

df = ingest_data("data/raw/adult.csv")
print("Columns:", df.columns)
print("Missing values per column:\n", df.isnull().sum())

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
print("X_train shape:", X_train.shape)  # Should be (26049, 14)
print("X_test shape:", X_test.shape)    # Should be (6513, 14)
