from src.data_preprocessing import X, Y
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random

# Load the model from the directory
feature_extractor = load_model("./final_model/custom_cnn_feature_extractor_finale.h5")
classification_model = joblib.load("./final_model/RF_model_finale.pkl")

def accuracy(Y_pred, Y):
    """
    This functions sums up all the correct prediction and divide it by total instances 
    to achieve total accuracy.
    """
    return np.sum( Y_pred == Y) / Y.size

def test_accuracy(X, Y):
    """
    This function uses the model to make prediction and return the accuracy as well as the predicted outcome.
    """
    features = feature_extractor.predict(X)
    Y_pred = classification_model.predict(features)
    return accuracy(Y_pred, Y), Y_pred

def predict(X):
    """
    This function predict a single instance and return the prediction as well as its confidence level.
    """
    features = feature_extractor.predict(X)
    Y_pred = classification_model.predict(features)
    probability = np.max(classification_model.predict_proba(features), axis=1)

    return Y_pred, probability

def test_prediction(index, X, Y):
    """
    This function is used to plot an image and its true label in conjunction with the prediction and its confidence level.
    """
    current = X[index]
    label = Y[index]
    current_image = np.expand_dims(current, axis=0)

    predicted_label, probability = predict(current_image)
    
    encode_label = {0:"Benign", 1:"Malignant", 2:"Normal"}

    print(f"Predicted label: {encode_label[int(predicted_label[0])]}")
    print(f"Actual label: {encode_label[int(label)]}")
    print(f"Confidence level: {probability}%")
  
    
    plt.imshow(current)
    plt.title(encode_label[int(predicted_label.item())])
    plt.axis("off")
    plt.show()

# Accuracy accross the testing set as well as its predicted outcome.
acc, Y_pred = test_accuracy(X, Y)

# The predicted outcome is used to generate a detailed reports.
print(f"Accuracy: {acc * 100}%")
print(classification_report(Y, Y_pred))

# randomly select 5 images and make a prediction on it.
for i in range(5): 
    index = random.randint(0, 9)
    test_prediction(index, X, Y)
