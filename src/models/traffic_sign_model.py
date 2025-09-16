from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import BatchNormalization, Activation
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from .interfaces import IModel

class BaseModel(IModel):
    """Base class for all models implementing the IModel interface."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
    
    def compile(self, optimizer: str = 'adam', loss: str = 'categorical_crossentropy',
                metrics: List[str] = None) -> None:
        """Compile the model with optimizer, loss function, and metrics.
        
        Args:
            optimizer: Name of optimizer or optimizer instance
            loss: Name of loss function or loss instance
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        metrics = metrics or ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def summary(self) -> None:
        """Print a summary of the model architecture."""
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
            
        self.model.summary()
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
            
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path: Path from where to load the model
        """
        self.model = load_model(path)


class TrafficSignCNN(BaseModel):
    """CNN model for traffic sign classification.
    
    Implements the IModel interface through BaseModel and follows the
    Single Responsibility Principle by focusing only on defining the model architecture.
    """
    
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> Sequential:
        """Build the CNN model architecture.
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of traffic sign classes
            
        Returns:
            Built Keras Sequential model
        """
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.15))
        
        # Second convolutional block
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.20))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))
        
        self.model = model
        return model


class TrafficSignResNet(BaseModel):
    """ResNet-style model for traffic sign classification.
    
    Implements the IModel interface through BaseModel and follows the
    Open/Closed Principle by extending the model capabilities without modifying existing code.
    """
    
    def _residual_block(self, x: Any, filters: int, kernel_size: int = 3, 
                       strides: int = 1, activation: str = 'relu') -> Any:
        """Create a residual block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Size of the kernel
            strides: Strides for the convolution
            activation: Activation function
            
        Returns:
            Output tensor
        """
        from tensorflow.keras.layers import add
        
        y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation(activation)(y)
        
        y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
        y = BatchNormalization()(y)
        
        if strides > 1:
            x = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
            x = BatchNormalization()(x)
        
        out = add([x, y])
        out = Activation(activation)(out)
        return out
    
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> Model:
        """Build the ResNet model architecture.
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of traffic sign classes
            
        Returns:
            Built Keras Model
        """
        inputs = Input(shape=input_shape)
        
        # Initial convolution
        x = Conv2D(32, kernel_size=5, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Residual blocks
        x = self._residual_block(x, 32)
        x = self._residual_block(x, 32)
        x = self._residual_block(x, 64, strides=2)
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 128, strides=2)
        x = self._residual_block(x, 128)
        
        # Output layers
        x = MaxPool2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.25)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model


class ModelFactory:
    """Factory for creating models.
    
    Follows the Factory Method pattern to create different types of models.
    """
    
    @staticmethod
    def create_model(model_type: str) -> IModel:
        """Create a model of the specified type.
        
        Args:
            model_type: Type of model to create ('cnn' or 'resnet')
            
        Returns:
            Instance of a model implementing IModel
            
        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type.lower() == 'cnn':
            return TrafficSignCNN()
        elif model_type.lower() == 'resnet':
            return TrafficSignResNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")