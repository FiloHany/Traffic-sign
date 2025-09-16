from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class IDataPreprocessor(ABC):
    """Interface for data preprocessors following the Interface Segregation Principle."""
    
    @abstractmethod
    def preprocess(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data and labels.
        
        Args:
            data: Input data
            labels: Input labels
            
        Returns:
            Tuple containing (preprocessed_data, preprocessed_labels)
        """
        pass


class IDataAugmenter(ABC):
    """Interface for data augmentation following the Interface Segregation Principle."""
    
    @abstractmethod
    def augment(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data and labels.
        
        Args:
            data: Input data
            labels: Input labels
            
        Returns:
            Tuple containing (augmented_data, augmented_labels)
        """
        pass


class IDataSplitter(ABC):
    """Interface for data splitting following the Interface Segregation Principle."""
    
    @abstractmethod
    def split(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data and labels into training and validation sets.
        
        Args:
            data: Input data
            labels: Input labels
            
        Returns:
            Dictionary containing split data and labels
        """
        pass