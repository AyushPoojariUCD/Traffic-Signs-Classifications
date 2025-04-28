import os
import logging
from dotenv import load_dotenv
from utils.logger import setup_logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.dataset_loader import load_dataset
from utils.preprocess_dataset import preprocess_data
from utils.dataset_extraction import extract_zip_file
from utils.download_datatsets import download_dataset
from pipeline.training import train_model
from pipeline.evaluation import evaluate_model
from pipeline.testing import predict_all_images,predict_image
from pipeline.visualization import visualize_evaluation_results

load_dotenv()

TRAFFIC_SIGN_PREPROCESSED_DATASET_LINK = os.getenv("TRAFFIC_SIGN_PREPROCESSED_DATASET_LINK")
TRAFFIC_SIGN_PREPROCESSED_DATASET_ZIP_FOLDER = os.getenv("TRAFFIC_SIGN_PREPROCESSED_DATASET_ZIP_FOLDER")

if __name__ == "__main__":

    setup_logging()


    # Dataset Loading
    logging.info("--------Downloading Dataset--------")
    download_dataset(TRAFFIC_SIGN_PREPROCESSED_DATASET_LINK)

    # Dataset Extraction
    logging.info("--------Extracting Dataset---------")
    extract_zip_file(TRAFFIC_SIGN_PREPROCESSED_DATASET_ZIP_FOLDER)

    # logging.info("Starting the  Pipeline")

    # Dataset Loading
    # logging.info("------Stage 1: Loading Dataset-----")
    # data = load_dataset("./traffic-signs-preprocessed-dataset/data3.pickle")

    # Preprocessing the dataset
    # logging.info("----Stage 2: Preprocessing Dataset----")
    # data = preprocess_data(data)

    # Training all models if needed or else artifact model
    models = ["cnn","lenet","alexnet","mcdnn","multiscale","vgg","resnet"]


    # Train models if needed: Uncomment if training is required
    '''
    for model in models:
        logging.info("Training All Model...")
        print(model)
        model, history, training_time = train_model(data, model_type=model, epochs=10, batch_size=128)

    
    logging.info(f"Model training completed")

    
    # Evaluate models if needed: Uncomment if evaluation is required
    # Evaluating the Model
    logging.info("Evaluating all models")
    for model in models:
        evaluate_model(
            model_type=model,
            model_path=f"./artifacts/{model}_model.h5",
            dataset_path="./traffic-signs-preprocessed-dataset/data3.pickle",
            num_classes=43
        )
    '''

    # As we have already trained and stored models in artifacts
    # We will just predict using models in artifacts
    logging.info("Testing images using trained models")
    results = predict_all_images()

    # Plot each image with its prediction
    for img_name, prediction in results.items():
        img_path = os.path.join('./test-images', img_name)

        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Prediction: {prediction}")
        plt.axis('off')
        plt.show()

    # Visualizing Model Performance
    # logging.info("Generating Visualizations...")
    # visualize_evaluation_results("./logs")

    logging.info("Pipeline Execution Completed Successfully!")





