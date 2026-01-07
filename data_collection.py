import pandas as pd
import re
import nltk

# --------------------------------------------------
# Safe NLTK Stopwords Download (only if not present)
# --------------------------------------------------
try:
    from nltk.corpus import stopwords
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
df = pd.read_csv("tweets.csv")

# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", str(text))
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)

# --------------------------------------------------
# Apply Cleaning
# --------------------------------------------------
df["Cleaned_Text"] = df["text"].apply(clean_text)

# --------------------------------------------------
# Save Cleaned Data
# --------------------------------------------------
df.to_csv("tweets_cleaned.csv", index=False)

print("CSV data cleaned successfully")
