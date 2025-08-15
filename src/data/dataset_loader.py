import numpy as np
from PIL import Image
import os

class DatasetLoader:
    """Loads and preprocesses dataset from local directory."""

    def __init__(self, dataset_dir: str, classes: int = 43, image_size=(30, 30)):
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.image_size = image_size

    def load_data(self):
        data, labels = [], []

        for class_id in range(self.classes):
            class_path = os.path.join(self.dataset_dir, 'train', str(class_id))
            images = os.listdir(class_path)

            for img_name in images:
                try:
                    image = Image.open(os.path.join(class_path, img_name))
                    image = image.resize(self.image_size)
                    image = np.array(image)
                    data.append(image)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

        return np.array(data), np.array(labels)