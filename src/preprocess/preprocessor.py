from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataPreprocessor:
    """Handles preprocessing of dataset."""

    @staticmethod
    def preprocess(data, labels, num_classes=43, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        return X_train, X_test, y_train, y_test