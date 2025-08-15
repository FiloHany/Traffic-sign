import kagglehub

class DatasetDownloader:
    """Handles downloading datasets from Kaggle."""
    
    @staticmethod
    def download(dataset_path: str) -> str:
        path = kagglehub.dataset_download(dataset_path)
        print("Path to dataset files:", path)
        return path