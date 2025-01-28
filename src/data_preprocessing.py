import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Define the base directory where images are stored
data_dir = "path\\to\\folder"

# Define class labels
classes = {'Bengin case', 'Malignant case', 'Normal case'}

# Initialize empty lists for images and labels
images = []
labels = []

image_size = 224

# Loop through each class directory
for class_name in classes:
    # Get the path to the current class directory
    class_dir = os.path.join(data_dir, class_name)

    # Loop through each image file in the class directory
    for filename in os.listdir(class_dir):
        # Get the full path to the image file
        image_path = os.path.join(class_dir, filename)

        # Load the image using OpenCV 
        image = cv2.imread(image_path)  # load the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert it RGB for 3 color channel
        image = cv2.resize(image, (image_size, image_size)) # resize to 224x224
        
        # Append the image and its corresponding label to the lists
        images.append(image)
        labels.append(class_name)

# Encode all the label to numeric values
label_dict = {"Bengin case": 0, "Malignant case": 1, "Normal case": 2}
encoded_labels = [label_dict[label] for label in labels]

# Generate a permutation of indices to perform random shuffling
permutation = np.random.permutation(len(encoded_labels))

X = np.array(images)
X = X / 255. # Normalize the pixel scale to 0-1 (0 = black, 1 = white)
Y = np.array(encoded_labels)

# shuffle images and labels
X = X[permutation]
Y = Y[permutation]




