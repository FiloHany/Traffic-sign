import tensorflow as tf

class ModelTrainer:
    """Handles model training and saving."""

    @staticmethod
    def train(model, X_train, y_train, X_test, y_test, epochs=25, batch_size=128):
        with tf.device('/GPU:0'):
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
        return history

    @staticmethod
    def save_model(model, path="traffic-sign-classifier.h5"):
        model.save(path)