import streamlit as st
import joblib
import sys
sys.path.append("../src")

from features import extract_features

model = joblib.load("models/url_model.pkl")

st.title("Phishing URL Detector 🔗")

url = st.text_input("Enter URL")

if st.button("Predict"):
    features = extract_features(url)
    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("🚨 Phishing Website")
    else:
        st.success("✅ Legitimate Website")