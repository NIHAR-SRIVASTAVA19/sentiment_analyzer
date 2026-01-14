import os
import pandas as pd
from utils import preprocess_text
from check_retrain import check_and_retrain

def log_update(test_log, correct_label):
    clean_text = preprocess_text(test_log)
    new_log = {"reviews": clean_text, "target": correct_label}
    new_df = pd.DataFrame([new_log])
    
    path = os.path.join('data', 'logs_file.csv')
    
    # Check if file exists to decide on the header
    file_exists = os.path.isfile(path)
    
    # mode='a' appends
    # header=not file_exists only writes the header the VERY FIRST time
    new_df.to_csv(path, mode='a', index=False, header=not file_exists)
    
    print(f"Logged: {clean_text} as {correct_label}")
    check_and_retrain()