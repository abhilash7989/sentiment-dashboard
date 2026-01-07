import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download VADER lexicon if not already
nltk.download('vader_lexicon')

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Hybrid prediction function
def hybrid_predict(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    # Check for unknown words
    unknown_words = [w for w in text.lower().split() if w not in vectorizer.vocabulary_]
    if unknown_words:
        score = sia.polarity_scores(text)["compound"]
        if score > 0.1:
            pred = "positive"
        elif score < -0.1:
            pred = "negative"
        else:
            pred = "neutral"
    return pred

# Streamlit UI
st.title("Social Media Sentiment Analysis Dashboard")
st.subheader("Analyze emotions from comments or tweets instantly")

option = st.radio("Choose Input Type:", ["Single Text", "Upload CSV"])

if option == "Single Text":
    text = st.text_area("Enter text here:")
    if st.button("Analyze Sentiment"):
        if text.strip():
            prediction = hybrid_predict(text)
            st.success(f"Predicted Sentiment: {prediction}")
        else:
            st.warning("Please enter some text.")

else:
    file = st.file_uploader("Upload CSV file with a 'Cleaned_Text' column", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        if "Cleaned_Text" not in data.columns:
            st.error("CSV must contain 'Cleaned_Text' column")
        else:
            data["Predicted"] = data["Cleaned_Text"].apply(hybrid_predict)
            st.write(data.head())
            
            # Plot sentiment distribution
            import plotly.express as px
            fig = px.pie(data, names="Predicted", title="Sentiment Distribution")
            st.plotly_chart(fig)
            
            fig_bar = px.bar(data["Predicted"].value_counts().reset_index(),
                             x="index", y="Predicted",
                             labels={"index": "Sentiment", "Predicted": "Count"},
                             title="Sentiment Counts")
            st.plotly_chart(fig_bar)
