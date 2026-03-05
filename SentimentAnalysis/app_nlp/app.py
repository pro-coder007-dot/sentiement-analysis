# app.py
import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load trained model + vectorizer
# ----------------------------
model, vectorizer = joblib.load("svm_model.pkl")

# ----------------------------
# Number to Emotion Mapping
# ----------------------------
NUMBER_TO_EMOTION = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

# Emoji Mapping
EMOJI_MAP = {
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😄"
}

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="ML-Based NLP Emotion Classifier",
    page_icon="😊",
    layout="wide"
)

st.title("😊 ML-Based NLP Emotion Classification Dashboard")
st.markdown("Classify emotions from text using a **Support Vector Machine (SVM)** model.")

# ----------------------------
# Sidebar: Model Info
# ----------------------------
st.sidebar.header("Model Info")
st.sidebar.write("**Type:** Support Vector Machine (SVM)")
st.sidebar.write("**Feature:** TF-IDF Vectorizer")
st.sidebar.write("**Frontend:** Streamlit")
st.sidebar.write("**Number of Emotions:** 6")
st.sidebar.write("**Accuracy:** ~88%")

# ----------------------------
# Input Section
# ----------------------------
st.subheader("Enter Text for Emotion Prediction")
user_text = st.text_area(
    "Type or paste text here:",
    height=150,
    placeholder="e.g., I'm feeling really sad today..."
)

predict_btn = st.button("Predict Emotion")

# ----------------------------
# Initialize history
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_btn and user_text.strip():
    # Preprocess text (optional: lowercase, strip)
    text_input = user_text.lower().strip()
    X_input = vectorizer.transform([text_input])
    
    # Predict emotion number
    predicted_number = model.predict(X_input)[0]
    
    # Map number to emotion text
    predicted_emotion = NUMBER_TO_EMOTION.get(predicted_number, "unknown")
    
    # Get confidence if available
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X_input)[0][predicted_number]
    
    # Save prediction in history
    st.session_state.history.append({
        "Text": user_text,
        "Emotion": predicted_emotion,
        "Emoji": EMOJI_MAP.get(predicted_emotion, "❓"),
        "Confidence": f"{confidence*100:.2f}%" if confidence else "N/A"
    })

# ----------------------------
# Show Prediction Result
# ----------------------------
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("📝 Latest Prediction")
    col1, col2, col3 = st.columns([2,1,1])
    col1.metric("Predicted Emotion", f"{latest['Emotion']} {latest['Emoji']}")
    col2.metric("Confidence", latest["Confidence"])
    col3.write("")  # Empty for spacing

# ----------------------------
# Prediction History
# ----------------------------
st.subheader("📊 Prediction History")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df[['Text', 'Emotion', 'Emoji', 'Confidence']], use_container_width=True)
else:
    st.write("No predictions yet.")