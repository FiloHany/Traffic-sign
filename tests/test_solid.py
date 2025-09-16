import sys
import os
import unittest
from pathlib import Path

# Add the src directory to the path so we can import modules
src_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(src_path))

from src.data.interfaces import IDatasetDownloader, IDatasetLoader, IDatasetInfo
from src.data.dataset_downloader import KaggleDatasetDownloader, LocalDatasetDownloader
from src.data.dataset_loader import TrafficSignDatasetLoader
from src.preprocess.interfaces import IDataPreprocessor, IDataAugmenter, IDataSplitter
from src.preprocess.preprocessor import StandardPreprocessor, OneHotEncoder, TrainTestSplitter, ImageAugmenter, PreprocessingPipeline
from src.models.interfaces import IModel, ITrainer, IEvaluator
from src.models.traffic_sign_model import ModelFactory, TrafficSignCNN, TrafficSignResNet
from src.models.trainer import ModelTrainer, ModelEvaluator, CallbackFactory
from src.utils.plot_utils import PlotterFactory, PerformancePlotter, ConfusionMatrixPlotter
from src.utils.datetime_utils import DateTimeFormatter, TimestampConverter
from src.main import TrafficSignRecognitionApp, create_app

class TestSOLIDPrinciples(unittest.TestCase):
    """Test cases to verify SOLID principles implementation."""
    
    def test_single_responsibility_principle(self):
        """Test that classes follow the Single Responsibility Principle."""
        # Each class should have a single responsibility
        downloader = KaggleDatasetDownloader()
        loader = TrafficSignDatasetLoader()
        preprocessor = StandardPreprocessor()
        encoder = OneHotEncoder()
        splitter = TrainTestSplitter()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        # Verify that each class has methods related to a single responsibility
        self.assertTrue(hasattr(downloader, 'download'))
        self.assertTrue(hasattr(loader, 'load_train_data'))
        self.assertTrue(hasattr(preprocessor, 'preprocess'))
        self.assertTrue(hasattr(encoder, 'encode'))
        self.assertTrue(hasattr(splitter, 'split'))
        self.assertTrue(hasattr(trainer, 'train'))
        self.assertTrue(hasattr(evaluator, 'evaluate'))
        
        # Verify that classes don't have methods that belong to other responsibilities
        self.assertFalse(hasattr(downloader, 'load_data'))
        self.assertFalse(hasattr(loader, 'preprocess'))
        self.assertFalse(hasattr(preprocessor, 'train'))
        self.assertFalse(hasattr(trainer, 'evaluate'))
    
    def test_open_closed_principle(self):
        """Test that classes follow the Open/Closed Principle."""
        # Create a preprocessing pipeline
        pipeline = PreprocessingPipeline()
        
        # Add components without modifying the pipeline class
        pipeline.add_preprocessor(StandardPreprocessor())
        pipeline.add_encoder(OneHotEncoder())
        pipeline.add_splitter(TrainTestSplitter())
        
        # Verify that the pipeline has the components
        self.assertEqual(len(pipeline.preprocessors), 1)
        self.assertEqual(len(pipeline.encoders), 1)
        self.assertEqual(len(pipeline.splitters), 1)
        
        # Add another preprocessor without modifying the pipeline class
        pipeline.add_augmenter(ImageAugmenter())
        self.assertEqual(len(pipeline.augmenters), 1)
    
    def test_liskov_substitution_principle(self):
        """Test that classes follow the Liskov Substitution Principle."""
        # Create a model factory
        factory = ModelFactory()
        
        # Create different model types
        cnn_model = factory.create_model('cnn', (32, 32, 3), 43)
        resnet_model = factory.create_model('resnet', (32, 32, 3), 43)
        
        # Both models should have the same interface
        self.assertTrue(hasattr(cnn_model, 'compile'))
        self.assertTrue(hasattr(cnn_model, 'fit'))
        self.assertTrue(hasattr(cnn_model, 'predict'))
        self.assertTrue(hasattr(cnn_model, 'evaluate'))
        
        self.assertTrue(hasattr(resnet_model, 'compile'))
        self.assertTrue(hasattr(resnet_model, 'fit'))
        self.assertTrue(hasattr(resnet_model, 'predict'))
        self.assertTrue(hasattr(resnet_model, 'evaluate'))
    
    def test_interface_segregation_principle(self):
        """Test that classes follow the Interface Segregation Principle."""
        # TrafficSignDatasetLoader implements both IDatasetLoader and IDatasetInfo
        loader = TrafficSignDatasetLoader()
        
        # Verify that it implements IDatasetLoader
        self.assertTrue(isinstance(loader, IDatasetLoader))
        self.assertTrue(hasattr(loader, 'load_train_data'))
        self.assertTrue(hasattr(loader, 'load_test_data'))
        
        # Verify that it implements IDatasetInfo
        self.assertTrue(isinstance(loader, IDatasetInfo))
        self.assertTrue(hasattr(loader, 'get_class_names'))
        self.assertTrue(hasattr(loader, 'get_class_distribution'))
        self.assertTrue(hasattr(loader, 'get_metadata'))
    
    def test_dependency_inversion_principle(self):
        """Test that classes follow the Dependency Inversion Principle."""
        # Create dependencies
        downloader = KaggleDatasetDownloader()
        loader = TrafficSignDatasetLoader()
        pipeline = PreprocessingPipeline()
        factory = ModelFactory()
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        # Create app with dependencies
        app = TrafficSignRecognitionApp(
            downloader=downloader,
            dataset_loader=loader,
            preprocessing_pipeline=pipeline,
            model_factory=factory,
            trainer=trainer,
            evaluator=evaluator
        )
        
        # Verify that app depends on abstractions, not concrete implementations
        self.assertTrue(isinstance(app.downloader, IDatasetDownloader))
        self.assertTrue(isinstance(app.dataset_loader, IDatasetLoader))
        self.assertTrue(isinstance(app.trainer, ITrainer))
        self.assertTrue(isinstance(app.evaluator, IEvaluator))
        
        # Create app with factory function
        app2 = create_app()
        
        # Verify that app2 also depends on abstractions
        self.assertTrue(isinstance(app2.downloader, IDatasetDownloader))
        self.assertTrue(isinstance(app2.dataset_loader, IDatasetLoader))
        self.assertTrue(isinstance(app2.trainer, ITrainer))
        self.assertTrue(isinstance(app2.evaluator, IEvaluator))

if __name__ == '__main__':
    unittest.main()