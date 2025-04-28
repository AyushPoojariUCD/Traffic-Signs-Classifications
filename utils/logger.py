import logging
import os
from datetime import datetime

def setup_logging():
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
            logging.StreamHandler()
        ]
    )
