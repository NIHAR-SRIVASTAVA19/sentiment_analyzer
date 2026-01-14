import pandas as pd
import os
from utils import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# DEFINE BALANCING LOGIC ---
def data_update(core_df, log_df):
    # Combine only the dataframes that actually exist
    to_concat = [df for df in [core_df, log_df] if df is not None]
    
    if not to_concat:
        raise ValueError("No data found to process! Check your CSV paths.")
        
    updated_data = pd.concat(to_concat, ignore_index=True)
    updated_data = updated_data.drop_duplicates(subset=['reviews', 'target'], keep='first')
    

    # Separate by target label
    df_neg = updated_data[updated_data['target'] == 0]
    df_neu = updated_data[updated_data['target'] == 1]
    df_pos = updated_data[updated_data['target'] == 2]

    # Sort by length to find the "Middle" count automatically
    # dfs[0]=min, dfs[1]=middle, dfs[2]=max
    dfs = sorted([df_neg, df_neu, df_pos], key=len)
    middle_count = len(dfs[1]) 

    balanced_dfs = []
    for df in dfs:
        # If len < middle, it upsamples. If len > middle, it downsamples.
        resampled = df.sample(n=middle_count, random_state=42, replace=(len(df) < middle_count))
        balanced_dfs.append(resampled)
    
    updated_data.to_csv(os.path.join('data','processed_amazon_review.csv'),index=False)
    return pd.concat(balanced_dfs, ignore_index=True)

