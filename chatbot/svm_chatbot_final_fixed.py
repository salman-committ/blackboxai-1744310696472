"""
Prompt Objective:
Train a machine learning-based chatbot using Support Vector Machine (SVM) that classifies user input into intents defined in a custom 'intents.json' file, and responds accordingly. The bot should:
- Preprocess text (lowercase, tokenization, stemming, remove stopwords)
- Use TfidfVectorizer for vectorization
- Train an SVM classifier
- Respond with randomized responses from matched intent
"""

# Import libraries
import json
import random
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # Ensure punkt_tab is downloaded

# Load intents JSON file
with open("data/training/intents.json") as file:
    data = json.load(file)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Function to preprocess input text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Prepare training data
texts = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(preprocess(pattern))
        labels.append(intent["tag"])

# Encode intent labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Build the SVM pipeline
model = make_pipeline(TfidfVectorizer(), SVC(kernel="linear", probability=True))
model.fit(texts, y)

# Prediction and response
def get_response(user_input):
    processed_input = preprocess(user_input)
    predicted_index = model.predict([processed_input])[0]
    intent_tag = label_encoder.inverse_transform([predicted_index])[0]

    for intent in data["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

# Chat loop
print("Yumaris Chatbot 🤖 — type 'quit' to exit.")
while True:
    inp = input("You: ")
    if inp.lower() in ["quit", "exit"]:
        print("Yumaris 🤖: Bye! Happy shopping 🛍️")
        break
    response = get_response(inp)
    print("Yumaris 🤖:", response)
