import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.preprocess_dataset import preprocess_data
from utils.dataset_loader import load_dataset

def evaluate_model(model_type="cnn", model_path = "./artifacts/cnn_model.h5",dataset_path="../traffic-signs-preprocessed-dataset/data3.pickle", num_classes=43):

    model = load_model(model_path)

    # Load and preprocess data
    data = load_dataset(dataset_path)
    data = preprocess_data(data)

    # Load training time
    training_time_path = f'./logs/{model_type}_training_time.txt'
    if os.path.exists(training_time_path):
        with open(training_time_path, 'r') as f:
            training_time = float(f.read().strip())
    else:
        training_time = None

    # Evaluate Validation Data
    val_loss, val_acc = model.evaluate(data['x_validation'], data['y_validation'], verbose=1)
    print(f'\nValidation Accuracy: {val_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')

    # Ensure y_test is categorical
    if data['y_test'].ndim == 1:
        from tensorflow.keras.utils import to_categorical
        data['y_test'] = to_categorical(data['y_test'], num_classes)

    # Evaluate Test Data
    test_loss, test_acc = model.evaluate(data['x_test'], data['y_test'], verbose=1)
    print(f'\nTest Accuracy: {test_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

    # Save Evaluation Metrics
    os.makedirs('./logs', exist_ok=True)
    model_results = [{
        "model_name": f"{model_type}_model",
        "validation_accuracy": float(val_acc),
        "validation_loss": float(val_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_time": float(training_time) if training_time is not None else None
    }]

    with open(f'./logs/{model_type}_evaluation.json', 'w') as f:
        json.dump(model_results, f, indent=4)

    print(f"\nEvaluation metrics saved to logs/{model_type}_evaluation.json")

    # 🚀 Added return
    return model_results
