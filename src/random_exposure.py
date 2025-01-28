import os
import random
from PIL import Image, ImageEnhance
import numpy as np

def random_exposure(image, min_exposure=0.8, max_exposure=1.2):
    """
    Apply random exposure adjustment to an image.
    :param image: PIL Image object
    :param min_exposure: Minimum exposure factor
    :param max_exposure: Maximum exposure factor
    :return: Exposure-adjusted image
    """
    enhancer = ImageEnhance.Brightness(image)
    exposure_factor = random.uniform(min_exposure, max_exposure)
    image = enhancer.enhance(exposure_factor)
    return image

def augment_image(image):
    """
    Apply various augmentations to an image, including random exposure.
    :param image: PIL Image object
    :return: Augmented image
    """
    image = random_exposure(image)

    return image

def process_images(input_folder, output_folder):
    """
    Process images from the input folder, apply augmentations, and save to the output folder.
    :param input_folder: Path to the input folder containing images
    :param output_folder: Path to the output folder to save augmented images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Apply augmentation
            augmented_image = augment_image(image)

            # Save the augmented image
            augmented_image.save(output_path)

            print(f'Processed and saved: {filename}')

# Define input and output folders
input_folder = 'path/to/folder'
output_folder = 'path/to/folder'

# Process the images
process_images(input_folder, output_folder)
