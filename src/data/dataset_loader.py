import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from .interfaces import IDatasetLoader, IDatasetInfo

class TrafficSignDatasetLoader(IDatasetLoader, IDatasetInfo):
    """Loads and preprocesses traffic sign dataset from local directory.
    
    Implements both IDatasetLoader and IDatasetInfo interfaces, following the
    Interface Segregation Principle by providing specific methods for loading
    and querying dataset information.
    """

    def __init__(self, dataset_dir: str, classes: int = 43, image_size: Tuple[int, int] = (30, 30)):
        """Initialize the dataset loader.
        
        Args:
            dataset_dir: Directory containing the dataset
            classes: Number of classes in the dataset
            image_size: Size to resize images to (width, height)
        """
        self.dataset_dir = dataset_dir
        self.classes = classes
        self.image_size = image_size
        self._metadata = None
        self._class_names = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from the dataset directory.
        
        Returns:
            Tuple containing (data, labels)
        """
        data, labels = [], []

        train_dir = os.path.join(self.dataset_dir, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found at {train_dir}")

        for class_id in range(self.classes):
            class_path = os.path.join(train_dir, str(class_id))
            if not os.path.exists(class_path):
                print(f"Warning: Class directory {class_id} not found, skipping")
                continue
                
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
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data from the dataset directory.
        
        Returns:
            Tuple containing (test_data, test_labels)
        """
        data, labels = [], []

        test_dir = os.path.join(self.dataset_dir, 'test')
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found at {test_dir}")

        # Check if there's a CSV file with test labels
        csv_path = os.path.join(self.dataset_dir, 'Test.csv')
        if os.path.exists(csv_path):
            # Load from CSV if available
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                try:
                    img_path = os.path.join(test_dir, row['Path'])
                    image = Image.open(img_path)
                    image = image.resize(self.image_size)
                    image = np.array(image)
                    data.append(image)
                    labels.append(row['ClassId'])
                except Exception as e:
                    print(f"Error loading test image {row['Path']}: {e}")
        else:
            # Otherwise, assume test directory has the same structure as train
            for class_id in range(self.classes):
                class_path = os.path.join(test_dir, str(class_id))
                if not os.path.exists(class_path):
                    continue
                    
                images = os.listdir(class_path)
                for img_name in images:
                    try:
                        image = Image.open(os.path.join(class_path, img_name))
                        image = image.resize(self.image_size)
                        image = np.array(image)
                        data.append(image)
                        labels.append(class_id)
                    except Exception as e:
                        print(f"Error loading test image {img_name}: {e}")

        return np.array(data), np.array(labels)
    
    def get_class_names(self) -> List[str]:
        """Get the names of classes in the dataset.
        
        Returns:
            List of class names
        """
        if self._class_names is not None:
            return self._class_names
            
        # Try to load from Meta.csv if available
        meta_path = os.path.join(self.dataset_dir, 'Meta.csv')
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            self._class_names = df['SignName'].tolist()
            return self._class_names
        
        # If no metadata file, return generic class names
        return [f"Class {i}" for i in range(self.classes)]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        _, labels = self.load_data()
        class_names = self.get_class_names()
        
        distribution = {}
        for i in range(self.classes):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            distribution[class_name] = np.sum(labels == i)
            
        return distribution
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset.
        
        Returns:
            Dictionary containing metadata
        """
        if self._metadata is not None:
            return self._metadata
            
        train_data, train_labels = self.load_data()
        test_data, test_labels = self.load_test_data()
        
        self._metadata = {
            'num_classes': self.classes,
            'image_size': self.image_size,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'color_channels': train_data.shape[3] if len(train_data.shape) > 3 else 1,
            'class_distribution': self.get_class_distribution()
        }
        
        return self._metadata