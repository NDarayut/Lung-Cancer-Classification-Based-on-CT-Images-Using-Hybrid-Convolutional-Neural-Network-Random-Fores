import os
import random
from PIL import Image, ImageOps
import numpy as np

def augment_image(image):
    # Horizontal flip with 50% chance of flipping occur
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # Random rotation
    angle = random.uniform(-10, 10)  # Rotate between -10 to 10 degrees
    image = image.rotate(angle)

    return image

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all the files in the input folder
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
input_folder = 'path/to/folder/data'
output_folder = 'path/to/folder/data'

# Process the images
process_images(input_folder, output_folder)
