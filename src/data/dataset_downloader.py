import kagglehub
import os
from pathlib import Path
from typing import Optional
from .interfaces import IDatasetDownloader

class KaggleDatasetDownloader(IDatasetDownloader):
    """Handles downloading datasets from Kaggle.
    
    Implements the IDatasetDownloader interface and follows the Single Responsibility Principle
    by focusing only on downloading datasets from Kaggle.
    """
    
    def download(self, dataset_path: str) -> str:
        """Download a dataset from Kaggle.
        
        Args:
            dataset_path: Kaggle dataset identifier (e.g., 'username/dataset-name')
            
        Returns:
            Local path where the dataset is stored
        """
        path = kagglehub.dataset_download(dataset_path)
        print(f"Dataset downloaded to: {path}")
        return path


class LocalDatasetDownloader(IDatasetDownloader):
    """Handles copying datasets from local directories.
    
    Implements the IDatasetDownloader interface for cases where the dataset
    is already available locally and just needs to be copied to the working directory.
    """
    
    def __init__(self, target_dir: Optional[str] = None):
        """Initialize the downloader with a target directory.
        
        Args:
            target_dir: Directory where datasets should be copied to.
                        If None, a default directory will be used.
        """
        self.target_dir = target_dir or os.path.join(os.getcwd(), 'data')
        os.makedirs(self.target_dir, exist_ok=True)
    
    def download(self, dataset_path: str) -> str:
        """Copy a dataset from a local path.
        
        Args:
            dataset_path: Path to the local dataset
            
        Returns:
            Path where the dataset is copied to
        """
        source_path = Path(dataset_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        # Create a subdirectory with the dataset name
        dataset_name = source_path.name
        target_path = Path(self.target_dir) / dataset_name
        
        # If it's a file, copy it; if it's a directory, copy its contents
        if source_path.is_file():
            os.makedirs(target_path.parent, exist_ok=True)
            import shutil
            shutil.copy2(source_path, target_path)
        else:
            import shutil
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        
        print(f"Dataset copied to: {target_path}")
        return str(target_path)