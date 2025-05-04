import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import sys

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # Explicitly download punkt_tab
    nltk.download('stopwords', quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    sys.exit(1)

# Function to preprocess text
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back to string
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return ""

# Load dataset
try:
    data = pd.read_csv('enron_spam_data.csv')
    print(f"Dataset loaded. Shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Handle missing values
data = data.dropna(subset=['text', 'label'])
if data.empty:
    print("Error: Dataset is empty after dropping missing values.")
    sys.exit(1)
print(f"Dataset after cleaning: {data.shape}")

# Preprocess email text
data['clean_text'] = data['text'].apply(preprocess_text)
print("Text preprocessing complete.")

# Encode labels (spam = 1, ham = 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
if data['label'].isnull().any():
    print("Error: Some labels could not be mapped. Check label values in dataset.")
    sys.exit(1)
print("Labels encoded.")

# Split features and target
X = data['clean_text']
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF conversion complete.")

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# Make predictions
y_pred = model.predict(X_test_tfidf)
print("Predictions complete.")

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Function to predict if a new email is spam
def predict_email(email_text):
    clean_email = preprocess_text(email_text)
    email_tfidf = vectorizer.transform([clean_email])
    prediction = model.predict(email_tfidf)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Example usage
sample_email = """
Subject: Win a Free Vacation!!!
Dear user, congratulations! You've won a free vacation to Hawaii. Click here to claim your prize now!
"""
print("\nSample Email Prediction:")
print(f"Email: {sample_email}")
print(f"Prediction: {predict_email(sample_email)}")