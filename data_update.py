import pandas as pd
import os

path=os.path.join("data","processed_amazon_review.csv")
try:
    if os.path.exists(path):
        data=pd.read_csv(path)
except Exception as e:
    print(f"Error reading the CSV file: {e}")

print(data.head())