import logging
import os

def setup_logger():
    """Set up the logger for the application."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Configure logger
    logger = logging.getLogger('chatbot')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler('logs/training.log')
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def clean_text(text):
    """Clean and preprocess text input."""
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def create_directories():
    """Create necessary directories for the project."""
    directories = ['data/training', 'data/models', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
