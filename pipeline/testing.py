import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Load the model once ---
model = load_model('./artifacts/vgg_model.h5')

# --- Load class labels ---
labels_df = pd.read_csv('./dataset-labels/label_names.csv')
class_labels = labels_df.set_index('ClassId')['SignName'].to_dict()


def load_and_prepare_image(img_file):
    """
    Load and preprocess a PIL image or file path for prediction.
    Accepts both file path (str) or uploaded file (BytesIO).
    """
    if isinstance(img_file, str):
        img = image.load_img(img_file, target_size=(32, 32))
    else:
        img = image.load_img(img_file, target_size=(32, 32))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
    img_array /= 255.0
    return img_array


def predict_image(img_file):
    """
    Predicts a single image.
    img_file: can be a file path or a BytesIO object (Streamlit Upload).
    """
    img_array = load_and_prepare_image(img_file)
    predictions = model.predict(img_array)
    predicted_class_id = np.argmax(predictions)
    predicted_class_name = class_labels.get(predicted_class_id, "Unknown")
    return predicted_class_name


def predict_all_images(test_images_dir='./test-images'):
    """
    Predicts all .png images in a directory.
    Returns a dictionary: {filename: prediction}.
    """
    results = {}
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_images_dir, filename)
            pred = predict_image(img_path)
            results[filename] = pred
    return results
