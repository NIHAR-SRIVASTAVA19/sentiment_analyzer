import streamlit as st
import joblib
import os
from log_update import log_update

# --- Load Model with Cache ---
model_path = os.path.join('model', 'sentiment_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

# --- Initialize Session State ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

st.title("Amazon Review Sentiment Analysis")
input_text = st.text_input("Enter your review here:", value=st.session_state.last_input)

if st.button("Predict Sentiment"):
    if input_text.strip():
        model = load_model()
        st.session_state.prediction = model.predict([input_text])[0]
        st.session_state.last_input = input_text
        st.session_state.feedback_submitted = False
    else:
        st.warning("Please enter a valid review text.")

# --- THE FEEDBACK LOGIC ---
if st.session_state.prediction is not None:
    sentiment_map = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}
    st.subheader(f"Result: {sentiment_map[st.session_state.prediction]}")

    if not st.session_state.feedback_submitted:
        # This checkbox controls the visibility of the feedback section
        is_incorrect = st.checkbox("Is the prediction incorrect?")
        
        if is_incorrect:
            st.write("### Provide Correct Feedback")
            correct_label = st.selectbox(
                "What was the correct sentiment?", 
                options=[0, 1, 2], 
                format_func=lambda x: sentiment_map[x]
            )
            
            if st.button("Submit Feedback"):
                log_update(st.session_state.last_input, correct_label)
                st.session_state.feedback_submitted = True
                st.rerun() # Refresh to update UI state
    else:
        st.success("✅ Feedback logged! This will help improve the model during the next batch update.")
        if st.button("Analyze New Review"):
            st.session_state.prediction = None
            st.session_state.feedback_submitted = False
            st.rerun()