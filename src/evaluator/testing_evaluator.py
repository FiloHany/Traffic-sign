from sklearn.metrics import accuracy_score

class ModelEvaluator:
    """Handles model evaluation (metrics calculation)."""
    @staticmethod
    def evaluate(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy with the test data: {accuracy}")
        return accuracy