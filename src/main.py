import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Import interfaces
from data.interfaces import IDatasetDownloader, IDatasetLoader, IDatasetInfo
from preprocess.interfaces import IDataPreprocessor, IDataAugmenter, IDataSplitter
from models.interfaces import IModel, ITrainer, IEvaluator

# Import implementations
from data.dataset_downloader import KaggleDatasetDownloader, LocalDatasetDownloader
from data.dataset_loader import TrafficSignDatasetLoader
from preprocess.preprocessor import StandardPreprocessor, OneHotEncoder, TrainTestSplitter, ImageAugmenter, PreprocessingPipeline
from models.traffic_sign_model import ModelFactory
from models.trainer import ModelTrainer, ModelEvaluator, CallbackFactory
from utils.plot_utils import PlotterFactory
from utils.datetime_utils import DateTimeFormatter

class TrafficSignRecognitionApp:
    """Main application class for Traffic Sign Recognition.
    
    This class follows the Dependency Inversion Principle by depending on abstractions
    (interfaces) rather than concrete implementations.
    """
    
    def __init__(self, 
                 downloader: IDatasetDownloader,
                 dataset_loader: IDatasetLoader,
                 preprocessing_pipeline: PreprocessingPipeline,
                 model_factory: ModelFactory,
                 trainer: ITrainer,
                 evaluator: IEvaluator):
        """Initialize the application with dependencies.
        
        Args:
            downloader: Dataset downloader implementation
            dataset_loader: Dataset loader implementation
            preprocessing_pipeline: Preprocessing pipeline
            model_factory: Model factory
            trainer: Model trainer implementation
            evaluator: Model evaluator implementation
        """
        self.downloader = downloader
        self.dataset_loader = dataset_loader
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model_factory = model_factory
        self.trainer = trainer
        self.evaluator = evaluator
        
        # Create output directories
        self.output_dir = Path("output")
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def run(self, 
            dataset_path: str,
            model_type: str = "cnn",
            epochs: int = 25,
            batch_size: int = 128,
            use_gpu: bool = True,
            use_augmentation: bool = True) -> Dict[str, Any]:
        """Run the complete traffic sign recognition pipeline.
        
        Args:
            dataset_path: Path to the dataset
            model_type: Type of model to use ('cnn' or 'resnet')
            epochs: Number of epochs for training
            batch_size: Batch size for training
            use_gpu: Whether to use GPU for training
            use_augmentation: Whether to use data augmentation
            
        Returns:
            Dictionary containing results and metrics
        """
        print(f"Starting Traffic Sign Recognition pipeline at {DateTimeFormatter.format()}")
        
        # Step 1: Download dataset if needed
        if not os.path.exists(dataset_path):
            print("Dataset not found. Downloading...")
            self.downloader.download(dataset_path)
        
        # Step 2: Load dataset
        print("Loading dataset...")
        X_train, y_train = self.dataset_loader.load_train_data(dataset_path)
        X_test, y_test = self.dataset_loader.load_test_data(dataset_path)
        
        # Get dataset info
        if isinstance(self.dataset_loader, IDatasetInfo):
            class_names = self.dataset_loader.get_class_names()
            class_distribution = self.dataset_loader.get_class_distribution()
            print(f"Dataset loaded with {len(class_names)} classes and {len(X_train)} training samples")
        else:
            class_names = None
            print(f"Dataset loaded with {X_train.shape[0]} training samples")
        
        # Step 3: Preprocess data
        print("Preprocessing data...")
        if use_augmentation:
            self.preprocessing_pipeline.add_augmenter(ImageAugmenter())
            
        X_train, X_val, y_train, y_val = self.preprocessing_pipeline.process(X_train, y_train, X_test, y_test)
        
        # Step 4: Build model
        print(f"Building {model_type} model...")
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]
        
        model = self.model_factory.create_model(model_type, input_shape, num_classes)
        
        # Step 5: Create callbacks
        callbacks = [
            CallbackFactory.create_early_stopping(patience=10),
            CallbackFactory.create_reduce_lr(patience=5),
            CallbackFactory.create_model_checkpoint(filepath=str(self.models_dir / "best_model.h5"))
        ]
        
        # Step 6: Train model
        print("Training model...")
        self.trainer.callbacks = callbacks
        history = self.trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        
        # Step 7: Evaluate model
        print("Evaluating model...")
        metrics = self.evaluator.evaluate(model, X_test, y_test)
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        # Step 8: Generate predictions
        y_pred = self.evaluator.predict(model, X_test)
        cm = self.evaluator.confusion_matrix(model, X_test, y_test)
        
        # Step 9: Plot results
        print("Plotting results...")
        performance_plotter = PlotterFactory.create_performance_plotter()
        performance_plotter.plot(history, save_path=str(self.plots_dir))
        
        confusion_plotter = PlotterFactory.create_confusion_matrix_plotter()
        confusion_plotter.plot({
            'confusion_matrix': cm,
            'class_names': class_names
        }, save_path=str(self.plots_dir))
        
        # Step 10: Save model
        model_path = str(self.models_dir / f"traffic_sign_{model_type}_model.h5")
        self.trainer.save_model(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Return results
        return {
            'metrics': metrics,
            'history': history,
            'model_path': model_path,
            'confusion_matrix': cm,
            'class_names': class_names
        }


def create_app(dataset_path: str = None) -> TrafficSignRecognitionApp:
    """Factory function to create the application with all dependencies.
    
    This function follows the Factory pattern to create the application with all its dependencies.
    
    Args:
        dataset_path: Path to the dataset, if None, a default path is used
        
    Returns:
        TrafficSignRecognitionApp instance
    """
    # Create dataset downloader
    downloader = KaggleDatasetDownloader()
    
    # Create dataset loader
    dataset_loader = TrafficSignDatasetLoader()
    
    # Create preprocessing pipeline
    preprocessor = StandardPreprocessor()
    encoder = OneHotEncoder()
    splitter = TrainTestSplitter(test_size=0.2)
    
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.add_preprocessor(preprocessor)
    preprocessing_pipeline.add_encoder(encoder)
    preprocessing_pipeline.add_splitter(splitter)
    
    # Create model factory
    model_factory = ModelFactory()
    
    # Create trainer and evaluator
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Create and return application
    return TrafficSignRecognitionApp(
        downloader=downloader,
        dataset_loader=dataset_loader,
        preprocessing_pipeline=preprocessing_pipeline,
        model_factory=model_factory,
        trainer=trainer,
        evaluator=evaluator
    )


if __name__ == "__main__":
    # Create application
    app = create_app()
    
    # Run application
    results = app.run(
        dataset_path="data/traffic_signs",
        model_type="resnet",  # or "cnn"
        epochs=25,
        batch_size=128,
        use_gpu=True,
        use_augmentation=True
    )
    
    print(f"Training completed with test accuracy: {results['metrics']['accuracy']:.4f}")