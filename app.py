import streamlit as st
import joblib
import os
import pandas as pd
from log_update import log_update
from config import MODEL_PATH, LOG_DATA_PATH, RETRAIN_THRESHOLD

# --- Page Configuration ---
st.set_page_config(page_title="Smart Sentiment AI", page_icon="📊", layout="centered")

# Corrected CSS block with 'unsafe_allow_html=True'
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #007bff; 
        color: white; 
    }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Loads and caches the ML model for better performance."""
    return joblib.load(MODEL_PATH)

# --- Sidebar: System Health Monitor ---
with st.sidebar:
    st.header("⚙️ System Status")
    if os.path.exists(LOG_DATA_PATH):
        try:
            logs = pd.read_csv(LOG_DATA_PATH)
            # Count only unique reviews to prevent spam from triggering retrain
            unique_logs = logs.drop_duplicates(subset=['reviews', 'target'])
            log_count = len(unique_logs)
            
            st.write(f"**Feedback Collected:** {log_count} / {RETRAIN_THRESHOLD}")
            st.progress(min(log_count / RETRAIN_THRESHOLD, 1.0))
            
            if log_count >= RETRAIN_THRESHOLD:
                st.warning("🔄 System is ready for batch retraining.")
        except Exception:
            st.info("Log file is currently being initialized.")
    else:
        st.info("No logs found yet.")
    
    st.divider()
    st.write("**Strategy:** Batch Learning (Periodic Retraining)")
    st.write("**Model:** Multinomial Naive Bayes")

# --- Main UI ---
st.title("🛒 Amazon Review Sentiment Analysis")
st.markdown("Enter a customer review below to classify it using our self-improving AI.")

# Initialize Session State
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Layout with columns for aligned input and button
col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

with col1:
    input_text = st.text_input("Customer Review:", value=st.session_state.last_input, placeholder="e.g., The product quality is amazing!")

with col2:
    predict_btn = st.button("Analyze")

if predict_btn:
    if input_text.strip():
        if os.path.exists(MODEL_PATH):
            model = load_model()
            # Predict and store in session state to persist through feedback
            st.session_state.prediction = model.predict([input_text])[0]
            st.session_state.last_input = input_text
            st.session_state.feedback_submitted = False
        else:
            st.error("Model not found! Please check your /model directory.")
    else:
        st.warning("Please enter some text before analyzing.")

# --- Results & Feedback Display ---
if st.session_state.prediction is not None:
    st.divider()
    sentiment_map = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}
    pred_label = sentiment_map[st.session_state.prediction]
    
    # Professional metric display
    st.metric(label="Predicted Sentiment", value=pred_label)

    if not st.session_state.feedback_submitted:
        # Use expander to keep UI clean
        with st.expander("Is this prediction wrong? Click to help the model learn!"):
            correct_label = st.selectbox(
                "What was the correct sentiment?",
                options=[0, 1, 2],
                format_func=lambda x: sentiment_map[x]
            )
            if st.button("Submit Correct Feedback"):
                log_update(st.session_state.last_input, correct_label)
                st.session_state.feedback_submitted = True
                st.rerun() # Refresh to update sidebar and show success message
    else:
        st.success("✅ Feedback logged! It will be integrated during the next update cycle.")
        if st.button("🔄 Analyze New Review"):
            st.session_state.prediction = None
            st.session_state.feedback_submitted = False
            st.session_state.last_input = ""
            st.rerun()