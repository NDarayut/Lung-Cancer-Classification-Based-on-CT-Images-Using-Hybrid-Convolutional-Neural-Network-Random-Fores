import cv2
import numpy as np
import logging
import psutil

def log_memory(stage):
    memory = psutil.virtual_memory()
    logging.info(f"{stage} - Memory usage: {memory.percent}%")

def preprocess(image_path):
    log_memory("Before loading image")
    try:
        # Define the size to which the image will be resized
        image_size = 224  # Reduce to 128x128 for memory efficiency

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Handle grayscale images
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        image = cv2.resize(image, (image_size, image_size))

        # Expand dimensions to match model input
        X = np.expand_dims(image, axis=0)

        # Normalize the pixel values to 0-1
        X = X / 255.0

        log_memory("After preprocessing")
        return X
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise
