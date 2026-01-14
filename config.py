import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
Actual_DATA_PATH = os.path.join(DATA_DIR, "amazon_review.csv")
CORE_DATA_PATH = os.path.join(DATA_DIR, "processed_amazon_review.csv")
LOG_DATA_PATH = os.path.join(DATA_DIR, "logs_file.csv")

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")

# Retraining Logic
RETRAIN_THRESHOLD = 10  # Retrain every 10 new reviews