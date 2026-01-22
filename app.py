import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="centered"
)

# --------------------------------------------------
# Load pretrained transformer model (cached)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )

classifier = load_model()

# Label mapping
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def is_valid_input(text):
    """Check if the text input is valid (3-500 chars)."""
    if text is None:
        return False
    text = text.strip()
    return 3 <= len(text) <= 500

def predict_sentiment(text):
    """Predict sentiment and return label + confidence."""
    result = classifier(text)[0]
    sentiment = label_map[result["label"]]
    confidence = result["score"]
    return sentiment, confidence

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.subheader("Powered by Pretrained Transformer Model (RoBERTa)")

option = st.radio(
    "Choose Input Type:",
    ["Single Text", "Upload CSV"],
    horizontal=True
)

# ---------------- Single Text Analysis ----------------
if option == "Single Text":
    text = st.text_area(
        "Enter text for sentiment analysis",
        placeholder="Type a tweet, comment, or review..."
    )

    if st.button("Analyze Sentiment"):
        if not is_valid_input(text):
            st.warning("Please enter valid text (3–500 characters).")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(text)

            if sentiment == "positive":
                st.success(f"😊 **Positive Sentiment**")
            elif sentiment == "negative":
                st.error(f"😠 **Negative Sentiment**")
            else:
                st.info(f"😐 **Neutral Sentiment**")

            st.write(f"**Confidence:** {confidence:.2f}")

# ---------------- CSV Upload Analysis ----------------
else:
    file = st.file_uploader(
        "Upload CSV file with a column named `text`",
        type=["csv"]
    )

    if file is not None:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("❌ CSV must contain a column named `text`")
        else:
            with st.spinner("Analyzing sentiments for uploaded data..."):
                df["Predicted_Sentiment"] = df["text"].astype(str).apply(
                    lambda x: predict_sentiment(x)[0]
                )

            st.success("✅ Sentiment analysis completed")
            st.dataframe(df.head())

            # ---------------- Pie Chart ----------------
            fig_pie = px.pie(
                df,
                names="Predicted_Sentiment",
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # ---------------- Bar Chart ----------------
            sentiment_counts = (
                df["Predicted_Sentiment"]
                .value_counts()
                .reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_bar = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Count",
                title="Sentiment Count",
                text="Count"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
