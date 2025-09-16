from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class IDatasetDownloader(ABC):
    """Interface for dataset downloaders following the Interface Segregation Principle."""
    
    @abstractmethod
    def download(self, dataset_path: str) -> str:
        """Download a dataset from a source.
        
        Args:
            dataset_path: Path or identifier for the dataset
            
        Returns:
            Local path where the dataset is stored
        """
        pass


class IDatasetLoader(ABC):
    """Interface for dataset loaders following the Interface Segregation Principle."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from a source.
        
        Returns:
            Tuple containing (data, labels)
        """
        pass
    
    @abstractmethod
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data from a source.
        
        Returns:
            Tuple containing (test_data, test_labels)
        """
        pass


class IDatasetInfo(ABC):
    """Interface for dataset metadata providers."""
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get the names of classes in the dataset.
        
        Returns:
            List of class names
        """
        pass
    
    @abstractmethod
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset.
        
        Returns:
            Dictionary containing metadata
        """
        pass