import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import os
import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.utils import clean_text, setup_logger

logger = setup_logger()

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
            
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise

    def fetch_product_details(self, product_name):
        """
        Fetch product details from Gemini AI.
        
        Args:
            product_name (str): The name of the product to fetch details for.
        
        Returns:
            dict: A dictionary containing the product picture URL and price.
        """
        try:
            from chatbot.gemini_config import configure_gemini, get_gemini_model, generate_product_prompt

            # Configure Gemini AI
            configure_gemini()
            model = get_gemini_model()

            # Generate prompt for product details
            prompt = generate_product_prompt(product_name)

            try:
                # Call Gemini AI to get product details
                response = model.generate_content(prompt)
                if response.text:
                    product_details = json.loads(response.text)
                else:
                    raise ValueError("Empty response from Gemini AI")
                
                # Validate the response structure
                required_fields = ['description', 'price', 'image_url']
                if not all(field in product_details for field in required_fields):
                    raise ValueError("Incomplete product details in response")
                
                return {
                    'description': product_details['description'],
                    'price': product_details['price'],
                    'image_url': product_details['image_url']
                }
            except json.JSONDecodeError:
                logger.error("Failed to parse Gemini AI response as JSON")
                return self._get_fallback_product_details()
            except Exception as e:
                logger.error(f"Error in Gemini AI request: {str(e)}")
                return self._get_fallback_product_details()
                
        except ImportError:
            logger.error("Gemini AI module not found")
            return self._get_fallback_product_details()
        except Exception as e:
            logger.error(f"Unexpected error in fetch_product_details: {str(e)}")
            return self._get_fallback_product_details()

    def _get_fallback_product_details(self):
        """Return fallback product details when Gemini AI fails"""
        return {
            'description': "Product details temporarily unavailable",
            'price': "Price information unavailable",
            'image_url': "https://via.placeholder.com/300x300.png?text=Product+Image+Unavailable"
        }
        return product_details

    def predict(self, text, include_media=False):
        """
        Predict the intent of the user's message and return an appropriate response.
        
        Args:
            text (str): The user's input text
            include_media (bool): Whether to include media handling capabilities in the response
            
        Returns:
            dict: Response containing text and media handling information
        """
        try:
            # Clean and vectorize the input text
            cleaned_text = clean_text(text)
            text_vector = self.vectorizer.fit_transform([cleaned_text]).toarray()
            
            # Get prediction
            prediction = self.model.predict(text_vector)
            predicted_class = self.encoder_classes[np.argmax(prediction)]
            
            # Find the corresponding intent
            for intent in self.intents['intents']:
                if intent['tag'] == predicted_class:
                    response = np.random.choice(intent['responses'])
                    
                    result = {
                        'text': response,
                        'intent': predicted_class,
                        'confidence': float(np.max(prediction))
                    }
                    
                    # Add media handling capabilities if requested
                    if include_media:
                        if predicted_class == 'photo_upload':
                            result['media_handler'] = {
                                'type': 'photo',
                                'accepted_formats': ['jpg', 'jpeg', 'png', 'gif'],
                                'max_size': 5242880  # 5MB
                            }
                        elif predicted_class == 'voice_message':
                            result['media_handler'] = {
                                'type': 'voice',
                                'accepted_formats': ['mp3', 'wav', 'ogg'],
                                'max_duration': 300  # 5 minutes
                            }
                    
                    return result
            
            # If no intent is found
            return {
                'text': "I'm not sure how to respond to that. Could you please rephrase?",
                'intent': 'unknown',
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'text': "I'm having trouble processing your request. Please try again.",
                'intent': 'error',
                'confidence': 0.0
            }

    def handle_media(self, media_type, media_data):
        """
        Handle uploaded media files.
        
        Args:
            media_type (str): Type of media ('photo' or 'voice')
            media_data (bytes): The media file data
            
        Returns:
            dict: Response indicating success or failure of media handling
        """
        try:
            if media_type == 'photo':
                # Here you would implement photo processing logic
                return {
                    'success': True,
                    'message': 'Photo received and processed successfully',
                    'media_type': 'photo'
                }
            elif media_type == 'voice':
                # Here you would implement voice processing logic
                return {
                    'success': True,
                    'message': 'Voice message received and processed successfully',
                    'media_type': 'voice'
                }
            else:
                return {
                    'success': False,
                    'message': f'Unsupported media type: {media_type}',
                    'media_type': media_type
                }
        except Exception as e:
            logger.error(f"Error handling media: {str(e)}")
            return {
                'success': False,
                'message': 'Error processing media file',
                'media_type': media_type
            }

if __name__ == "__main__":
    # Example usage
    chatbot = ChatbotPredictor()
    
    # Test text prediction
    test_messages = [
        "Hello there!",
        "I want to upload a photo",
        "Can I send a voice message?",
        "Goodbye"
    ]
    
    print("Testing chatbot responses:\n")
    for message in test_messages:
        response = chatbot.predict(message, include_media=True)
        print(f"User: {message}")
        print(f"Bot: {response['text']}")
        if 'media_handler' in response:
            print(f"Media handling: {response['media_handler']}")
        print(f"Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
        print()
