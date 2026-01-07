import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER if not present
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Load cleaned data
df = pd.read_csv("tweets_cleaned.csv")

# Label sentiment with stricter thresholds
def get_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"

df["Sentiment"] = df["Cleaned_Text"].apply(get_sentiment)

# Check distribution
print(df["Sentiment"].value_counts())

# Save labeled dataset
df.to_csv("tweets_labeled.csv", index=False)
print("tweets_labeled.csv created successfully")
