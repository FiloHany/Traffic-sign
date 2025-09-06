import numpy as np
import tensorflow as tf

class ModelPredictor:
    """Handles prediction using the trained model."""
    def __init__(self, model):
        self.model = model

    def predict_classes(self, X):
        with tf.device('/GPU:0'):
            predictions = np.argmax(self.model.predict(X), axis=-1)
        return predictions