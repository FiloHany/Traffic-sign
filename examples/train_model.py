import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import modules
src_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(src_path))

from src.main import create_app

def main():
    """Example script to train a traffic sign recognition model."""
    # Create the application with default dependencies
    app = create_app()
    
    # Define dataset path - adjust as needed
    dataset_path = "../data/traffic_signs"
    
    # Run the application with custom parameters
    results = app.run(
        dataset_path=dataset_path,
        model_type="cnn",  # Use "cnn" or "resnet"
        epochs=20,
        batch_size=64,
        use_gpu=True,
        use_augmentation=True
    )
    
    # Print results
    print("\nTraining Results:")
    print(f"Test accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Test loss: {results['metrics']['loss']:.4f}")
    print(f"Model saved to: {results['model_path']}")
    
    # You can access other results like:
    # - results['history']: Training history
    # - results['confusion_matrix']: Confusion matrix
    # - results['class_names']: Class names

if __name__ == "__main__":
    main()