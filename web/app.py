# For this to work, you must install: python==3.11
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model 
from image_preprocess import preprocess
import os
import numpy as np
import joblib

# Load model from folders
feature_extractor = load_model(r"../final_model/custom_cnn_feature_extractor_finale.h5")
classification_model = joblib.load(r"../final_model/RF_model_finale.pkl")

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['POST', 'GET'])

def predict():
    # If client visit page, return only the html file
    if request.method == 'GET':
        return render_template('index.html')
        
    # If client submit an image, we perform preprocessing and classification
    elif request.method == 'POST':
        imagefile = request.files['imagefile']

        # Prevent use from submitting without actually input image
        if imagefile.filename == '':
            return render_template('index.html', prediction='No file selected', image=None)

        # Save images into image folders
        image_path = os.path.join(app.root_path, 'static\images', imagefile.filename)
        imagefile.save(image_path)

        # preprocess image e.g np array and scaling
        X = preprocess(image_path)
        # declare encoder for each target
        encode_label = {0:"Benign", 1:"Malignant", 2:"Normal"}
        
        # extract image feature
        features = feature_extractor.predict(X)
        # reshape it for Random Forest 
        features = features.reshape(features.shape[0], -1)

        # Prediction on Random Forest Classifier
        Y_pred = classification_model.predict(features)
        # Get all prediction accross the entire Forest and calculate confidence level
        probability = np.max(classification_model.predict_proba(features), axis=1)*100
        Y_pred_index = int(Y_pred[0])  # Extract the first element if Y_pred is an array
        probability_value = float(probability[0])  # Extract the first element of the probabilities
        classification = '%s (%.2f%%)' % (encode_label[Y_pred_index], probability_value)
        
        # Create a bar chart of probabilities
        probabilities = classification_model.predict_proba(features)[0] * 100  # Assuming single image prediction
        classes = list(encode_label.values())
        
        # Set backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Create and save the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, probabilities, color=['blue', 'red', 'green'])
        # Add percentage labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')  # Adjust the y-position as needed
        plt.ylim(0, max(probabilities) + 10)  # Add a little padding above the highest bar
        plt.xlabel('Classes')
        plt.ylabel('Confidence Level (%)')
        plt.title('Confidence Level for Each Class')
        
        # Save the plot to the static/images folder
        chart_path = os.path.join(app.root_path, 'static', 'images', f'chart.png')
        plt.savefig(chart_path)
        plt.close()  # Close the plot to free memory

        # Pass the result, the submitted image and the chart image to HTML
        return render_template('index.html', prediction=classification, image=imagefile.filename, chart=f'images/chart.png')

if __name__ == '__main__':
    # Runs on port 3000
    app.run(port=3000, debug =True)

