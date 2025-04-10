import sys
import os

# Add the chatbot directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.predict import ChatbotPredictor

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
