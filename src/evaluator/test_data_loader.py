import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

class TestDataLoader:
    """Handles loading and preprocessing of test dataset."""
    def __init__(self, csv_path, img_base_path, img_size=(30, 30)):
        self.csv_path = csv_path
        self.img_base_path = img_base_path
        self.img_size = img_size

    def load_labels_and_paths(self):
        """Load labels and image paths from CSV."""
        df = pd.read_csv(self.csv_path)
        labels = df["ClassId"].values
        img_paths = df["Path"].values
        return labels, img_paths

    def preprocess_images(self, img_paths):
        """Load and preprocess images for model input."""
        data = []
        with tf.device('/GPU:0'):
            for path in img_paths:
                image = Image.open(f"{self.img_base_path}/{path}")
                image = image.resize(self.img_size)
                data.append(np.array(image))
        return np.array(data)

    def load_test_data(self):
        """Complete pipeline: load labels, paths, and preprocessed images."""
        labels, img_paths = self.load_labels_and_paths()
        X_test = self.preprocess_images(img_paths)
        return X_test, labels