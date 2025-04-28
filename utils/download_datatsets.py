import os
import gdown
from dotenv import load_dotenv

load_dotenv()

def download_dataset(file_id):

    if not file_id:
        raise ValueError("Dataset link not found in environment variables.")

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, quiet=False)
    print("Dataset download completed")

