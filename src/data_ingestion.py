import yaml
from src.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["raw_path"]

def load_data(path: str):
    logger.info(f"Reading data from {path}")
    df = pd.read_csv(path, header=0, skipinitialspace=True)
    logger.info(f"Data shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(df.shape)
    print(df.head())
