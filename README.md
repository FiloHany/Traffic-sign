# Traffic Sign Recognition System

## Project Overview
This project implements a Traffic Sign Recognition system using deep learning techniques. The system is designed to classify traffic signs from images, which is a critical component for autonomous driving systems and driver assistance technologies.

## SOLID Principles Implementation
The project has been refactored to follow SOLID principles:

1. **Single Responsibility Principle (SRP)**
   - Each class has a single responsibility
   - Example: `TrafficSignDatasetLoader` is responsible only for loading data, while `ModelTrainer` is responsible only for training models

2. **Open/Closed Principle (OCP)**
   - Classes are open for extension but closed for modification
   - Example: `PreprocessingPipeline` can be extended with new preprocessors without modifying its code

3. **Liskov Substitution Principle (LSP)**
   - Subtypes can be substituted for their base types
   - Example: `TrafficSignCNN` and `TrafficSignResNet` can be used interchangeably where `IModel` is expected

4. **Interface Segregation Principle (ISP)**
   - Clients should not be forced to depend on interfaces they do not use
   - Example: Separate interfaces like `IDatasetDownloader`, `IDatasetLoader`, and `IDatasetInfo`

5. **Dependency Inversion Principle (DIP)**
   - High-level modules depend on abstractions, not concrete implementations
   - Example: `TrafficSignRecognitionApp` depends on interfaces like `IDatasetDownloader` and `ITrainer`

## Completed Components
- Project structure with SOLID principles
- Data infrastructure (downloaders, loaders, interfaces)
- Preprocessing pipeline with modular components
- Model architecture with dependency injection
- Training and evaluation modules
- Visualization utilities
- Example scripts

## Technical Stack
- Python 3.8+
- TensorFlow/Keras for deep learning
- NumPy, Pandas for data manipulation
- Matplotlib, Seaborn for visualization
- Scikit-learn for metrics and utilities

## Project Structure
```
Traffic_Sign/
├── Source/
│   ├── data/                  # Dataset storage
│   ├── examples/              # Example scripts
│   ├── output/                # Output directory for models and plots
│   ├── src/
│   │   ├── data/              # Data handling modules
│   │   │   ├── interfaces.py  # Data interfaces
│   │   │   ├── dataset_downloader.py
│   │   │   └── dataset_loader.py
│   │   ├── models/            # Model architecture and training
│   │   │   ├── interfaces.py  # Model interfaces
│   │   │   ├── traffic_sign_model.py
│   │   │   └── trainer.py
│   │   ├── preprocess/        # Data preprocessing
│   │   │   ├── interfaces.py  # Preprocessing interfaces
│   │   │   └── preprocessor.py
│   │   ├── utils/             # Utility functions
│   │   │   ├── datetime_utils.py
│   │   │   └── plot_utils.py
│   │   ├── notebooks/         # Jupyter notebooks
│   │   └── main.py            # Main application
│   ├── requirements.txt       # Project dependencies
│   └── README.md              # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Training a Model
```python
from src.main import create_app

# Create application with dependencies
app = create_app()

# Run the training pipeline
results = app.run(
    dataset_path="data/traffic_signs",
    model_type="resnet",  # or "cnn"
    epochs=25,
    batch_size=128,
    use_gpu=True,
    use_augmentation=True
)

# Access results
print(f"Test accuracy: {results['metrics']['accuracy']:.4f}")
```

Alternatively, run the example script:
```
python examples/train_model.py
```

#### Making Predictions
```python
import tensorflow as tf
from src.models.trainer import ModelEvaluator
from src.data.dataset_loader import TrafficSignDatasetLoader

# Load model
model = tf.keras.models.load_model("output/models/traffic_sign_model.h5")

# Create evaluator
evaluator = ModelEvaluator()

# Make predictions
predictions = evaluator.predict(model, X_test)
```

Alternatively, run the example script:
```
python examples/predict.py
```

## Contributors
- Traffic Sign Recognition Team

## License
This project is licensed under the MIT License - see the LICENSE file for details.
