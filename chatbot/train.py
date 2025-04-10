import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.model import ChatbotModel
from chatbot.utils import setup_logger, clean_text, create_directories
from chatbot.config import DATASET_PATH, MODEL_SAVE_PATH

# Set up logger
logger = setup_logger()

def load_training_data():
    """Load and process training data from intents.json."""
    try:
        with open(DATASET_PATH, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"Training data file not found at {DATASET_PATH}")
        raise
    except json.JSONDecodeError:
        logger.error("Error parsing training data JSON file")
        raise

def prepare_training_data(data):
    """Prepare training data for the model."""
    patterns = []
    tags = []
    
    # Extract patterns and tags
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(clean_text(pattern))
            tags.append(intent['tag'])
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(patterns).toarray()
    
    # Save vectorizer for later use
    if not os.path.exists('data/models'):
        os.makedirs('data/models')
    np.save('data/models/vectorizer.npy', vectorizer.vocabulary_)
    
    return X, tags, vectorizer

def main():
    """Main training function."""
    try:
        # Create necessary directories
        create_directories()
        
        # Load and prepare data
        logger.info("Loading training data...")
        data = load_training_data()
        
        logger.info("Preparing training data...")
        X, tags, vectorizer = prepare_training_data(data)
        
        # Initialize model
        model = ChatbotModel()
        
        # Encode labels
        logger.info("Encoding labels...")
        model.encoder.fit(tags)
        y = model.encoder.transform(tags)
        
        # Save encoder classes for later use
        np.save('data/models/encoder_classes.npy', model.encoder.classes_)
        
        # Split the data
        logger.info("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train the model
        logger.info("Building and training the model...")
        model.build_model(input_shape=X_train.shape[1])
        model.train(X_train, y_train)
        
        # Save the model
        logger.info("Saving the trained model...")
        if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
        model.save_model(MODEL_SAVE_PATH)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
