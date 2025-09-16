import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
from .datetime_utils import DateTimeFormatter

class IPlotter(ABC):
    """Interface for plotting functionality.
    
    Follows the Interface Segregation Principle by defining a specific plotting interface.
    """
    
    @abstractmethod
    def plot(self, data: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot the data and optionally save to file.
        
        Args:
            data: Data to plot
            save_path: Path to save the plot, if None, the plot is displayed
        """
        pass

class IMultiPlotter(ABC):
    """Interface for plotting multiple plots.
    
    Follows the Interface Segregation Principle by defining a specific multi-plotting interface.
    """
    
    @abstractmethod
    def plot_multiple(self, data_list: List[Dict[str, Any]], 
                     layout: Tuple[int, int], 
                     save_path: Optional[str] = None) -> None:
        """Plot multiple data sets and optionally save to file.
        
        Args:
            data_list: List of data to plot
            layout: Tuple of (rows, cols) for subplot layout
            save_path: Path to save the plot, if None, the plot is displayed
        """
        pass

class PerformancePlotter(IPlotter):
    """Plots model performance metrics.
    
    Implements the IPlotter interface and follows the Single Responsibility Principle
    by focusing only on plotting model performance.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        """Initialize the plotter with figure size.
        
        Args:
            figsize: Size of the figure as (width, height)
        """
        self.figsize = figsize
    
    def plot(self, history: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot training history and optionally save to file.
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save the plot, if None, the plot is displayed
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot accuracy
        self._plot_subplot(
            ax=ax1,
            train_metric=history['accuracy'],
            val_metric=history.get('val_accuracy', None),
            title='Model Accuracy',
            ylabel='Accuracy',
            xlabel='Epoch'
        )
        
        # Plot loss
        self._plot_subplot(
            ax=ax2,
            train_metric=history['loss'],
            val_metric=history.get('val_loss', None),
            title='Model Loss',
            ylabel='Loss',
            xlabel='Epoch'
        )
        
        plt.tight_layout()
        
        if save_path:
            timestamp = DateTimeFormatter.format(option='filename')
            full_path = f"{save_path}/performance_{timestamp}.png"
            plt.savefig(full_path)
            plt.close()
            print(f"Plot saved to {full_path}")
        else:
            plt.show()
    
    def _plot_subplot(self, ax: plt.Axes, train_metric: List[float], 
                     val_metric: Optional[List[float]], title: str, 
                     ylabel: str, xlabel: str) -> None:
        """Plot a subplot with training and validation metrics.
        
        Args:
            ax: Matplotlib axes to plot on
            train_metric: Training metric values
            val_metric: Validation metric values
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
        """
        epochs = range(1, len(train_metric) + 1)
        
        ax.plot(epochs, train_metric, 'b-', label='Training')
        if val_metric:
            ax.plot(epochs, val_metric, 'r-', label='Validation')
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

class ConfusionMatrixPlotter(IPlotter):
    """Plots confusion matrices.
    
    Implements the IPlotter interface and follows the Single Responsibility Principle
    by focusing only on plotting confusion matrices.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize the plotter with figure size.
        
        Args:
            figsize: Size of the figure as (width, height)
        """
        self.figsize = figsize
    
    def plot(self, data: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot confusion matrix and optionally save to file.
        
        Args:
            data: Dictionary containing 'confusion_matrix' and optionally 'class_names'
            save_path: Path to save the plot, if None, the plot is displayed
        """
        import seaborn as sns
        
        cm = data['confusion_matrix']
        class_names = data.get('class_names', None)
        
        plt.figure(figsize=self.figsize)
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            timestamp = DateTimeFormatter.format(option='filename')
            full_path = f"{save_path}/confusion_matrix_{timestamp}.png"
            plt.savefig(full_path)
            plt.close()
            print(f"Plot saved to {full_path}")
        else:
            plt.show()

class ImageGridPlotter(IPlotter):
    """Plots a grid of images.
    
    Implements the IPlotter interface and follows the Single Responsibility Principle
    by focusing only on plotting image grids.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 12)):
        """Initialize the plotter with figure size.
        
        Args:
            figsize: Size of the figure as (width, height)
        """
        self.figsize = figsize
    
    def plot(self, data: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot a grid of images and optionally save to file.
        
        Args:
            data: Dictionary containing 'images', 'labels', and optionally 'predictions' and 'class_names'
            save_path: Path to save the plot, if None, the plot is displayed
        """
        images = data['images']
        labels = data['labels']
        predictions = data.get('predictions', None)
        class_names = data.get('class_names', None)
        
        # Determine grid size based on number of images
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        plt.figure(figsize=self.figsize)
        
        for i in range(min(n_images, grid_size * grid_size)):
            plt.subplot(grid_size, grid_size, i + 1)
            
            # Handle grayscale images
            if len(images[i].shape) == 2 or images[i].shape[-1] == 1:
                plt.imshow(images[i].squeeze(), cmap='gray')
            else:
                plt.imshow(images[i])
            
            # Create title with true label and prediction if available
            title = f"True: {labels[i]}"
            if class_names:
                title = f"True: {class_names[labels[i]]}"
                
            if predictions is not None:
                pred_label = predictions[i]
                if class_names:
                    pred_label = class_names[predictions[i]]
                title += f"\nPred: {pred_label}"
                
                # Add color to indicate correct/incorrect prediction
                if predictions[i] == labels[i]:
                    plt.title(title, color='green')
                else:
                    plt.title(title, color='red')
            else:
                plt.title(title)
                
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            timestamp = DateTimeFormatter.format(option='filename')
            full_path = f"{save_path}/image_grid_{timestamp}.png"
            plt.savefig(full_path)
            plt.close()
            print(f"Plot saved to {full_path}")
        else:
            plt.show()

class PlotterFactory:
    """Factory for creating different types of plotters.
    
    Follows the Factory Method pattern to create different types of plotters.
    """
    
    @staticmethod
    def create_performance_plotter(figsize: Tuple[int, int] = (12, 5)) -> PerformancePlotter:
        """Create a performance plotter.
        
        Args:
            figsize: Size of the figure as (width, height)
            
        Returns:
            PerformancePlotter instance
        """
        return PerformancePlotter(figsize=figsize)
    
    @staticmethod
    def create_confusion_matrix_plotter(figsize: Tuple[int, int] = (10, 8)) -> ConfusionMatrixPlotter:
        """Create a confusion matrix plotter.
        
        Args:
            figsize: Size of the figure as (width, height)
            
        Returns:
            ConfusionMatrixPlotter instance
        """
        return ConfusionMatrixPlotter(figsize=figsize)
    
    @staticmethod
    def create_image_grid_plotter(figsize: Tuple[int, int] = (12, 12)) -> ImageGridPlotter:
        """Create an image grid plotter.
        
        Args:
            figsize: Size of the figure as (width, height)
            
        Returns:
            ImageGridPlotter instance
        """
        return ImageGridPlotter(figsize=figsize)