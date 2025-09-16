import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# Add the src directory to the path so we can import modules
src_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(src_path))

from src.data.dataset_loader import TrafficSignDatasetLoader
from src.models.trainer import ModelEvaluator
from src.utils.plot_utils import PlotterFactory

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for model input
    """
    # Load image
    img = Image.open(image_path)
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def main():
    """Example script to use a trained model for prediction."""
    # Path to the saved model
    model_path = "../output/models/traffic_sign_cnn_model.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Create dataset loader to get class names
    dataset_loader = TrafficSignDatasetLoader()
    dataset_path = "../data/traffic_signs"
    
    # Check if dataset exists
    if os.path.exists(dataset_path):
        class_names = dataset_loader.get_class_names(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}. Class names will not be available.")
        class_names = None
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Path to test images
    test_images_dir = "../data/test_images"
    
    # Check if test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found at {test_images_dir}.")
        print("Using sample data from the dataset instead.")
        
        # Load some test data from the dataset
        if os.path.exists(dataset_path):
            X_test, y_test = dataset_loader.load_test_data(dataset_path)
            
            # Select a few random samples
            indices = np.random.choice(len(X_test), size=9, replace=False)
            images = X_test[indices]
            labels = np.argmax(y_test[indices], axis=1)
            
            # Make predictions
            predictions = evaluator.predict_classes(model, images)
            
            # Plot results
            plotter = PlotterFactory.create_image_grid_plotter()
            plotter.plot({
                'images': images,
                'labels': labels,
                'predictions': predictions,
                'class_names': class_names
            })
        else:
            print("Neither test images nor dataset found. Cannot make predictions.")
        return
    
    # Get all image files in the test directory
    image_files = [f for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {test_images_dir}")
        return
    
    # Process each image
    images = []
    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        img_array = load_and_preprocess_image(image_path)
        images.append(img_array[0])  # Remove batch dimension for plotting
    
    # Stack images for batch prediction
    batch_images = np.stack([load_and_preprocess_image(os.path.join(test_images_dir, f))[0] 
                           for f in image_files])
    
    # Make predictions
    predictions = evaluator.predict_classes(model, batch_images)
    
    # Print predictions
    print("\nPredictions:")
    for i, (image_file, prediction) in enumerate(zip(image_files, predictions)):
        class_name = class_names[prediction] if class_names else f"Class {prediction}"
        print(f"{i+1}. {image_file}: {class_name} (Class {prediction})")
    
    # Plot results
    plotter = PlotterFactory.create_image_grid_plotter()
    plotter.plot({
        'images': np.array(images),
        'labels': np.zeros(len(images), dtype=int),  # Placeholder labels
        'predictions': predictions,
        'class_names': class_names
    })

if __name__ == "__main__":
    main()