import streamlit as st
import joblib
import pandas as pd
from src.features import extract_features

model = joblib.load("models/url_model.pkl")

st.title("Phishing URL Detector")
st.markdown("Enter a URL to analyze")

url = st.text_input("Enter URL")

if st.button("Predict"):
    if url:
        features = extract_features(url)

        # 🔹 Prediction
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"Phishing Website")
        else:
            st.success(f"Legitimate Website")

        st.subheader("Probability Scores")
        st.write(f"Phishing Probability: **{proba:.2f}**")
        st.write(f"Legitimate Probability: **{1 - proba:.2f}**")

        if proba > 0.7:
            st.warning("⚠️ High Risk URL")
        elif proba > 0.4:
            st.info("⚠️ Medium Risk URL")
        else:
            st.success("Safe URL")

        st.subheader("🔍 Extracted Features")

        feature_names = [
            "URL Length", "Dots", "Hyphens", "@ Symbol", "HTTPS",
            "Slashes", "Queries", "Equals",
            "Digits Count", "Digit Ratio",
            "Special Char Count", "Special Char Ratio",
            "IP Address", "Suspicious Words",
            "Domain Length", "Subdomains", "Domain Has Digits",
            "Path Length", "Subdirectories",
            "Shortened URL", "Suspicious TLD",
            "Entropy", "Longest Word", "Avg Word Length",
            "Double Slash", "HTTP", "Repeating Chars",
            "Encoded Chars", "Length Difference"
        ]

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": features
        })

        st.dataframe(feature_df)

        st.subheader("Feature Importance")

        importance = model.get_feature_importance()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    else:
        st.warning("Please enter a URL")