import os
import pickle

# Loading dataset
def load_dataset(dataset_path='../traffic-signs-preprocessed-dataset/data3.pickle'):
    """
    Loading preprocessed traffic sign dataset from a pickle file.

    Args: dataset_path (str): Path to the preprocessed dataset pickle file.

    Returns: dict: Dictionary containing 'x_train', 'y_train', 'x_validation', 'y_validation', etc.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    with open(dataset_path, 'rb') as dataset:
        data = pickle.load(dataset, encoding='latin1')

    return data
