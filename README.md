# 🤖Smart Amazon Review Sentiment Analyser (Self-Improving AI)

A professional **sentiment analysis application** built with **Python**, **Scikit-Learn**, and **Streamlit**.  
Unlike static models, this system features an automated **Feedback Loop** that allows the model to **learn and retrain itself** based on real-world user corrections.

---

## 🌟 Key Features

- **Real-time Prediction**  
  Classifies Amazon reviews as:
  - Positive 🟢  
  - Neutral 🟡  
  - Negative 🔴  

- **Active Learning**  
  Users can correct the model when it makes an incorrect prediction.

- **Automated Batch Retraining**  
  The system monitors a **Unique Feedback Threshold** (default: `10`) and automatically triggers a retraining cycle.

- **Dynamic Data Balancing**  
  Implements a custom **Middle-Class Sampling Strategy** to prevent sentiment bias.

- **Data Persistence**  
  All human-verified feedback is archived into the core dataset, gradually forming a **Golden Dataset**.

---

## 🏗️ Project Structure

```plaintext
SENTIMENT_ANALYSER/
├── data/
│   ├── amazon_review.csv            # Original raw dataset
│   ├── processed_amazon_review.csv  # Core dataset (grows over time)
│   └── logs_file.csv                # Temporary feedback storage
├── model/
│   └── sentiment_model.pkl          # Trained ML pipeline ("Brain")
├── app.py                           # Streamlit web interface
├── check_retrain.py                 # Automated retraining trigger logic
├── data_update.py                   # Data balancing & cleaning logic
├── log_update.py                    # Feedback ingestion script
├── train_sentiments.ipynb           # Model training & experimentation notebook
├── config.py                        # Global configuration & paths
├── utils.py                         # Text preprocessing utilities
└── requirements.txt                 # Project dependencies
```
## 🚀 Getting Started
**1️⃣ Installation**

Clone the repository and install the required dependencies:

```plaintext
pip install -r requirements.txt
```
---

**2️⃣ Run the Application**

Start the Streamlit web server:

```plaintext
streamlit run app.py
```
---

**3️⃣ Self-Improving Feedback Loop**

- Enter an Amazon review and click Predict

- If the prediction is incorrect, check “Is the prediction incorrect?”

- Select the correct sentiment and click Submit

- After 10 unique corrections, the system automatically retrains

Terminal output:

```Threshold reached. Starting retraining...```

---

## 🧠 Technical Highlights
**📊 Dynamic Balancing Logic**

To avoid bias toward any sentiment class, the data_update.py script:

- Identifies the median class size

- Oversamples minority classes

- Undersamples majority classes
This ensures stable and balanced learning during retraining.
---

**🔁 Deduplication**

During every retraining cycle:

- drop_duplicates() is applied automatically

- Prevents repeated or spam feedback from degrading model performance
---

**🛠️ Technologies Used**

- Python 3.13

- Pandas – Data manipulation and deduplication

- Scikit-Learn – TF-IDF vectorization & Logistic Regression

- Streamlit – Interactive web interface and session state

- Joblib – Model persistence
---

## 📌 Future Enhancements

- Online / incremental learning

- Transformer-based sentiment models

- User confidence scoring

- Feedback analytics dashboard
---

⭐ If you find this project useful, consider starring the repository!


