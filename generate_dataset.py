import pandas as pd
import random

# Positive, negative, neutral sentences
positive_sentences = [
    "I love this product", "This is amazing", "Absolutely fantastic experience",
    "Very happy with the service", "Best decision ever", "Great quality and performance",
    "Highly recommended", "Excellent support team", "Super satisfied", "Works perfectly"
]

negative_sentences = [
    "I hate this", "Very bad experience", "Terrible product",
    "Extremely disappointed", "Waste of money", "Poor quality",
    "Not recommended", "Worst service ever", "Completely useless", "Very unhappy"
]

neutral_sentences = [
    "This is okay", "Average experience", "Nothing special",
    "It works", "Fine for now", "Acceptable quality",
    "Normal product", "No strong opinion", "Just okay", "As expected"
]

# Generate balanced dataset
data = []
for _ in range(1000):
    data.append([random.choice(positive_sentences)])
    data.append([random.choice(negative_sentences)])
    data.append([random.choice(neutral_sentences)])

df = pd.DataFrame(data, columns=["Cleaned_Text"])

# Save raw dataset
df.to_csv("tweets_cleaned.csv", index=False)
print("tweets_cleaned.csv created successfully")
