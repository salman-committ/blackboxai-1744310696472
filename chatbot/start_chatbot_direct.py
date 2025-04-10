import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import os
import logging

# Add the chatbot directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.utils import clean_text, setup_logger

class ChatbotPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.encoder_classes = None
        self.intents = None
        self.initialize()

    def initialize(self):
        """Initialize the chatbot by loading the model and required files."""
        try:
            # Load the trained model
            model_path = os.path.join('data', 'models', 'chatbot_model.h5')
            self.model = load_model(model_path)
            
            # Load the vectorizer vocabulary
            vectorizer_path = os.path.join('data', 'models', 'vectorizer.npy')
            vocab = np.load(vectorizer_path, allow_pickle=True).item()
            self.vectorizer = TfidfVectorizer(vocabulary=vocab, max_features=1000)
            
            # Load encoder classes
            encoder_classes_path = os.path.join('data', 'models', 'encoder_classes.npy')
            self.encoder_classes = np.load(encoder_classes_path)
            
            # Load intents
            intents_path = os.path.join('data', 'training', 'intents.json')
            with open(intents_path, 'r') as f:
                self.intents = json.load(f)
            
            logging.info("Chatbot initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing chatbot: {str(e)}")
            raise

    def predict(self, text):
        """Predict the intent of the user's message and return an appropriate response."""
        try:
            cleaned_text = clean_text(text)
            text_vector = self.vectorizer.transform([cleaned_text]).toarray()
            prediction = self.model.predict(text_vector)
            predicted_class = self.encoder_classes[np.argmax(prediction)]
            
            for intent in self.intents['intents']:
                if intent['tag'] == predicted_class:
                    response = np.random.choice(intent['responses'])
                    return {'text': response, 'intent': predicted_class}
            
            return {'text': "I'm not sure how to respond to that.", 'intent': 'unknown'}
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return {'text': "I'm having trouble processing your request.", 'intent': 'error'}

def start_chatbot():
    """Start the chatbot and test its response."""
    try:
        chatbot = ChatbotPredictor()  # Initialize the chatbot
        test_message = "Hello, how can I help you?"  # Sample input
        response = chatbot.predict(test_message)  # Get response
        print(f"Chatbot response: {response['text']}")  # Print response
    except Exception as e:
        print(f"Error starting chatbot: {str(e)}")

if __name__ == "__main__":
    start_chatbot()
