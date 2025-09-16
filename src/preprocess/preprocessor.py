from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .interfaces import IDataPreprocessor, IDataSplitter, IDataAugmenter

class StandardPreprocessor(IDataPreprocessor):
    """Handles standard preprocessing of image data.
    
    Implements the IDataPreprocessor interface and follows the Single Responsibility Principle
    by focusing only on basic preprocessing operations.
    """
    
    def __init__(self, normalize: bool = True):
        """Initialize the preprocessor.
        
        Args:
            normalize: Whether to normalize the data
        """
        self.normalize = normalize
    
    def preprocess(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data by normalizing and reshaping if needed.
        
        Args:
            data: Input image data
            labels: Input labels
            
        Returns:
            Tuple containing (preprocessed_data, labels)
        """
        # Ensure data is float32 for better performance
        data = data.astype('float32')
        
        # Normalize if requested
        if self.normalize:
            data = data / 255.0
            
        return data, labels


class OneHotEncoder(IDataPreprocessor):
    """Handles one-hot encoding of labels.
    
    Implements the IDataPreprocessor interface and follows the Single Responsibility Principle
    by focusing only on encoding labels.
    """
    
    def __init__(self, num_classes: int = 43):
        """Initialize the encoder.
        
        Args:
            num_classes: Number of classes for one-hot encoding
        """
        self.num_classes = num_classes
    
    def preprocess(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert labels to one-hot encoded format.
        
        Args:
            data: Input data (unchanged)
            labels: Input labels to be one-hot encoded
            
        Returns:
            Tuple containing (data, one_hot_labels)
        """
        one_hot_labels = to_categorical(labels, self.num_classes)
        return data, one_hot_labels


class TrainTestSplitter(IDataSplitter):
    """Handles splitting data into training and test sets.
    
    Implements the IDataSplitter interface and follows the Single Responsibility Principle
    by focusing only on data splitting.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the splitter.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data and labels into training and test sets.
        
        Args:
            data: Input data
            labels: Input labels
            
        Returns:
            Dictionary containing X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=self.test_size, random_state=self.random_state
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


class ImageAugmenter(IDataAugmenter):
    """Handles image augmentation for training data.
    
    Implements the IDataAugmenter interface and follows the Single Responsibility Principle
    by focusing only on data augmentation.
    """
    
    def __init__(self, rotation_range: int = 10, width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1, zoom_range: float = 0.1,
                 horizontal_flip: bool = False):
        """Initialize the augmenter with transformation parameters.
        
        Args:
            rotation_range: Degree range for random rotations
            width_shift_range: Fraction of width for random horizontal shifts
            height_shift_range: Fraction of height for random vertical shifts
            zoom_range: Range for random zoom
            horizontal_flip: Whether to randomly flip images horizontally
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        
    def augment(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data by applying random transformations.
        
        Args:
            data: Input image data
            labels: Input labels
            
        Returns:
            Tuple containing (augmented_data, augmented_labels)
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Create an image data generator
        datagen = ImageDataGenerator(
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip
        )
        
        # Fit the generator on the data
        datagen.fit(data)
        
        # For simplicity, we'll just generate a fixed number of augmented samples
        # In a real application, you might use flow() to generate batches during training
        augmented_data = []
        augmented_labels = []
        
        # Generate augmented samples for each class
        for i in range(len(data)):
            # Generate 3 augmented versions of each image
            for _ in range(3):
                img = data[i].reshape((1,) + data[i].shape)
                for batch in datagen.flow(img, batch_size=1):
                    augmented_data.append(batch[0])
                    augmented_labels.append(labels[i])
                    break
        
        # Combine original and augmented data
        combined_data = np.vstack([data, np.array(augmented_data)])
        combined_labels = np.hstack([labels, np.array(augmented_labels)])
        
        return combined_data, combined_labels


class PreprocessingPipeline:
    """Combines multiple preprocessors into a pipeline.
    
    Follows the Composite pattern to combine multiple preprocessing steps.
    """
    
    def __init__(self, preprocessors: List[IDataPreprocessor] = None,
                 augmenter: Optional[IDataAugmenter] = None,
                 splitter: Optional[IDataSplitter] = None):
        """Initialize the pipeline with preprocessors, augmenter, and splitter.
        
        Args:
            preprocessors: List of preprocessors to apply in sequence
            augmenter: Optional augmenter to apply after preprocessing
            splitter: Optional splitter to split data after preprocessing and augmentation
        """
        self.preprocessors = preprocessors or []
        self.augmenter = augmenter
        self.splitter = splitter
    
    def process(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Process data through the pipeline.
        
        Args:
            data: Input data
            labels: Input labels
            
        Returns:
            Dictionary containing processed data and labels
        """
        # Apply each preprocessor in sequence
        current_data, current_labels = data, labels
        for preprocessor in self.preprocessors:
            current_data, current_labels = preprocessor.preprocess(current_data, current_labels)
        
        # Apply augmentation if available
        if self.augmenter:
            augmented_data, augmented_labels = self.augmenter.augment(current_data, current_labels)
            current_data, current_labels = augmented_data, augmented_labels
        
        # Split data if splitter is available
        if self.splitter:
            return self.splitter.split(current_data, current_labels)
        
        # Otherwise, return the processed data and labels
        return {
            'X': current_data,
            'y': current_labels
        }