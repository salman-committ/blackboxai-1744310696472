# Configuration file for the e-commerce chatbot

import os

# Dataset file path
DATASET_PATH = os.path.join("data", "training", "intents.json")

# Model save path
MODEL_SAVE_PATH = os.path.join("data", "models", "chatbot_model.h5")

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32

# Logging configuration
LOG_FILE_PATH = os.path.join("logs", "training.log")
