from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class IModel(ABC):
    """Interface for machine learning models following the Interface Segregation Principle."""
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> Any:
        """Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes
            
        Returns:
            Built model
        """
        pass
    
    @abstractmethod
    def compile(self, optimizer: str = 'adam', loss: str = 'categorical_crossentropy',
                metrics: List[str] = None) -> None:
        """Compile the model with optimizer, loss function, and metrics.
        
        Args:
            optimizer: Name of optimizer or optimizer instance
            loss: Name of loss function or loss instance
            metrics: List of metrics to track
        """
        pass
    
    @abstractmethod
    def summary(self) -> None:
        """Print a summary of the model architecture."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path: Path from where to load the model
        """
        pass


class ITrainer(ABC):
    """Interface for model trainers following the Interface Segregation Principle."""
    
    @abstractmethod
    def train(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the model on data.
        
        Args:
            model: Model to train
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        pass


class IEvaluator(ABC):
    """Interface for model evaluators following the Interface Segregation Principle."""
    
    @abstractmethod
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            model: Model to evaluate
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model.
        
        Args:
            model: Model to use for prediction
            X: Input data
            
        Returns:
            Predictions
        """
        pass