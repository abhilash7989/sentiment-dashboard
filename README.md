# Sentiment Analysis Dashboard

## Overview
This project is a Sentiment Analysis Dashboard built in Python using Streamlit. It allows users to input text (tweets, reviews, etc.) and predicts the sentiment as Positive, Negative, or Neutral. The project includes data preprocessing, model training using TF-IDF + Logistic Regression, and a web interface for easy interaction.

## Features
1.Clean and preprocess text data (removes URLs, special characters, stopwords).
2.Train a sentiment analysis model on labeled tweets.
3.Save and reuse trained model and vectorizer using `joblib`.
4.Streamlit dashboard for predicting sentiment of any text.
5.Works offline with a generated sample dataset if real data is unavailable.

## File Structure
sentiment-analysis-dashboard/
│
├─ data_collection.py # Optional: Script to fetch tweets from Twitter API
├─ generate_dataset.py # Script to create dataset (tweets.csv)
├─ preprocessing.py # Script to clean and label the data
├─ train_model.py # Script to train Logistic Regression model
├─ app.py # Streamlit dashboard
├─ requirements.txt # Required Python packages
├─ sentiment_model.pkl # Trained model (generated after training)
├─ vectorizer.pkl # TF-IDF vectorizer (generated after training)
├─ tweets.csv # Original dataset
├─ tweets_cleaned.csv # Cleaned dataset
├─ tweets_labeled.csv # Labeled dataset
└─ README.md # Project documentation


---

## Installation

1. Clone the repository**

**bash
git clone https://github.com/<your-username>/sentiment-dashboard.git
cd sentiment-dashboard

2.create a virtual environment
**bash
python -m venv venv
# Activate venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

3.Install dependencies
**bash
pip install -r requirements.txt

4.usage

1.generate dataset
2.preprocess dataset
3.train the model
4.run the streamlit dashboard

AUTHOR:ABHILASH
LINKDIN:www.linkedin.com/in/abhilash-abhi-b7415a25a
EMAIL:abhilashabhi4246@gmail.com
