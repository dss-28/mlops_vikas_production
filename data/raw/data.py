import pandas as pd

# Path to your downloaded file
txt_path = r"C:\Users\dss28\OneDrive\Desktop\mlops\mlops_vikas\data\raw\adult.data.txt"
csv_path = r"C:\Users\dss28\OneDrive\Desktop\mlops\mlops_vikas\data\raw\adult.csv"

# Column names as per UCI Adult dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race",
    "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

# Read the TXT file
df = pd.read_csv(txt_path, names=columns, sep=", ", engine='python')

# Save as CSV
df.to_csv(csv_path, index=False)

print("Converted to CSV:", csv_path)
print(df.head())
