import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from .interfaces import ITrainer, IEvaluator, IModel

class ModelTrainer(ITrainer):
    """Handles model training with configurable callbacks.
    
    Implements the ITrainer interface and follows the Single Responsibility Principle
    by focusing only on training models.
    """
    
    def __init__(self, callbacks: List[tf.keras.callbacks.Callback] = None):
        """Initialize the trainer with optional callbacks.
        
        Args:
            callbacks: List of Keras callbacks to use during training
        """
        self.callbacks = callbacks or []
    
    def train(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 25, batch_size: int = 128,
              use_gpu: bool = True) -> Dict[str, Any]:
        """Train the model on data.
        
        Args:
            model: Model to train
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs to train
            batch_size: Batch size for training
            use_gpu: Whether to use GPU for training
            
        Returns:
            Dictionary containing training history
        """
        # Set up validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Use GPU if requested and available
        if use_gpu and tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=self.callbacks
                )
        else:
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=self.callbacks
            )
        
        return history.history
    
    def save_model(self, model: Any, path: str = "traffic-sign-classifier.h5") -> None:
        """Save the trained model to disk.
        
        Args:
            model: Trained model to save
            path: Path where to save the model
        """
        model.save(path)


class ModelEvaluator(IEvaluator):
    """Handles model evaluation and prediction.
    
    Implements the IEvaluator interface and follows the Single Responsibility Principle
    by focusing only on evaluating models.
    """
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            model: Model to evaluate
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = model.evaluate(X_test, y_test, verbose=0)
        
        # Create a dictionary of metric names and values
        metrics = {}
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = float(results[i])
        
        return metrics
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model.
        
        Args:
            model: Model to use for prediction
            X: Input data
            
        Returns:
            Predictions
        """
        return model.predict(X)
    
    def predict_classes(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Predict class indices for input data.
        
        Args:
            model: Model to use for prediction
            X: Input data
            
        Returns:
            Predicted class indices
        """
        predictions = self.predict(model, X)
        return np.argmax(predictions, axis=1)
    
    def confusion_matrix(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Compute confusion matrix for the model.
        
        Args:
            model: Model to evaluate
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        
        # Convert one-hot encoded labels to class indices
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
            
        # Predict classes
        y_pred = self.predict_classes(model, X_test)
        
        # Compute confusion matrix
        return confusion_matrix(y_true, y_pred)


class CallbackFactory:
    """Factory for creating Keras callbacks.
    
    Follows the Factory Method pattern to create different types of callbacks.
    """
    
    @staticmethod
    def create_early_stopping(monitor: str = 'val_loss', patience: int = 10,
                             restore_best_weights: bool = True) -> tf.keras.callbacks.EarlyStopping:
        """Create an early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will be stopped
            restore_best_weights: Whether to restore model weights from the epoch with the best value
            
        Returns:
            EarlyStopping callback
        """
        return tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
    
    @staticmethod
    def create_model_checkpoint(filepath: str, monitor: str = 'val_loss',
                               save_best_only: bool = True) -> tf.keras.callbacks.ModelCheckpoint:
        """Create a model checkpoint callback.
        
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model
            
        Returns:
            ModelCheckpoint callback
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only
        )
    
    @staticmethod
    def create_reduce_lr(monitor: str = 'val_loss', factor: float = 0.2,
                        patience: int = 5) -> tf.keras.callbacks.ReduceLROnPlateau:
        """Create a reduce learning rate callback.
        
        Args:
            monitor: Metric to monitor
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which learning rate will be reduced
            
        Returns:
            ReduceLROnPlateau callback
        """
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=1e-6
        )
    
    @staticmethod
    def create_tensorboard(log_dir: str) -> tf.keras.callbacks.TensorBoard:
        """Create a TensorBoard callback.
        
        Args:
            log_dir: Directory where to save the log files
            
        Returns:
            TensorBoard callback
        """
        return tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )