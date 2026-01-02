import pandas as pd

# Path to your CSV
path = r"C:\Users\dss28\OneDrive\Desktop\mlops\mlops_vikas\data\raw\adult.csv"

# Read CSV without assuming header
df = pd.read_csv(path, header=None)
print("Shape of CSV:", df.shape)
print("First 5 rows:\n", df.head())
