import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import json
import os

class ChatbotModel:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()

    def build_model(self, input_shape):
        """Build the neural network model."""
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.encoder.classes_), activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    def train(self, X, y):
        """Train the model."""
        self.model.fit(X, y, epochs=100, batch_size=32)

    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)

    def load_model(self, path):
        """Load a trained model."""
        self.model = load_model(path)
