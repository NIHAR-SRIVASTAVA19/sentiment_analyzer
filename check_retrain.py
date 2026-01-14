import pandas as pd
import os
import joblib
from config import CORE_DATA_PATH, LOG_DATA_PATH, MODEL_PATH,RETRAIN_THRESHOLD as THRESHOLD
from data_update import data_update # Importing the balancing logic we built
from utils import preprocess_text
from sklearn.model_selection import train_test_split

def check_and_retrain():
    #  Check if logs exist and count rows
    if not os.path.exists(LOG_DATA_PATH):
        print("No logs found. Skipping update.")
        return

    log_df = pd.read_csv(LOG_DATA_PATH)
    # Count only UNIQUE reviews in the log
    unique_log_count = len(log_df.drop_duplicates(subset=['reviews', 'target']))

    #  Compare the UNIQUE count to the threshold
    if unique_log_count < THRESHOLD:
        print(f"Unique logs: {unique_log_count}/{THRESHOLD}. Still waiting for fresh data...")
        return

    print(f"Threshold reached with {unique_log_count} unique logs. Retraining...")
    #  Load Core Data
    core_df = pd.read_csv(CORE_DATA_PATH)

    #  Use your Strategy A balancing function
    balanced_data = data_update(core_df, log_df)
    balanced_data['reviews'] = balanced_data['reviews'].apply(preprocess_text)

    #  Train the Model
    # We load the existing pipeline to keep the same Vectorizer settings
    model = joblib.load(MODEL_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_data['reviews'], 
        balanced_data['target'], 
        test_size=0.2, 
        random_state=42, 
        stratify=balanced_data['target']
    )

    model.fit(X_train, y_train)
    
    #  Save the updated model
    joblib.dump(model, MODEL_PATH)
    print("✅ Model updated successfully!")

    #  CLEAR THE LOGS [The important part!]
    # We overwrite the file with only the column headers
    empty_df = pd.DataFrame(columns=['reviews', 'target'])
    empty_df.to_csv(LOG_DATA_PATH, index=False)
    print("🗑️ logs_file.csv has been cleared for the next batch.")

if __name__ == "__main__":
    check_and_retrain()