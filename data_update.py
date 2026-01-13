import pandas as pd
import os
from utils import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# --- STEP 1: LOAD DATA SAFELY ---
core_path = os.path.join("data", "processed_amazon_review.csv")
log_path = os.path.join("data", "logs_file.csv")

core_data = pd.read_csv(core_path) if os.path.exists(core_path) else None
log_data = pd.read_csv(log_path) if os.path.exists(log_path) else None

# --- STEP 2: LOAD EXISTING MODEL ---
model_path = os.path.join('model', 'sentiments_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("Warning: Original model not found. Ensure you have trained it once first.")

# --- STEP 3: DEFINE BALANCING LOGIC ---
def data_update(core_df, log_df):
    # Combine only the dataframes that actually exist
    to_concat = [df for df in [core_df, log_df] if df is not None]
    
    if not to_concat:
        raise ValueError("No data found to process! Check your CSV paths.")
        
    updated_data = pd.concat(to_concat, ignore_index=True)

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

    return pd.concat(balanced_dfs, ignore_index=True)

# --- STEP 4: EXECUTE UPDATE ---
# 1. Get balanced data
balanced_data = data_update(core_data, log_data)

# 2. Preprocess text (The 'apply' method is the professional way to do this)
balanced_data['reviews'] = balanced_data['reviews'].apply(preprocess_text)

# 3. Train-Test Split (Ensuring we have a test set to verify the refresh)
x_train, x_test, y_train, y_test = train_test_split(
    balanced_data['reviews'], 
    balanced_data['target'], 
    test_size=0.2, 
    random_state=42, 
    stratify=balanced_data['target']
)

# 4. Re-fit the model on the fresh data
model.fit(x_train, y_train)

# 5. Check accuracy
y_pred = model.predict(x_test)
print(f"Refreshed Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 6. SAVE THE IMPROVED MODEL
joblib.dump(model, model_path)
print("Model updated and saved successfully!")