from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from image_preprocess import preprocess
import os
import numpy as np
import joblib
import threading
import time

# Load model from folders
feature_extractor = load_model(r"../final_model/custom_cnn_feature_extractor_finale.h5")
classification_model = joblib.load(r"../final_model/RF_model_finale.pkl")

app = Flask(__name__, static_url_path='/static')

def delete_files_after_delay(files, delay=10):
    """
    Delete files after a specified delay.
    :param files: List of file paths to delete.
    :param delay: Time in seconds to wait before deleting files.
    """
    time.sleep(delay)
    for file in files:
        if os.path.exists(file):
            os.remove(file)

@app.route('/', methods=['POST', 'GET'])
def predict():
    # If client visit page, return only the HTML file
    if request.method == 'GET':
        return render_template('index.html')

    # If client submit an image, perform preprocessing and classification
    elif request.method == 'POST':
        imagefile = request.files['imagefile']

        # Prevent submission without an image
        if imagefile.filename == '':
            return render_template('index.html', prediction='No file selected', image=None)

        # Save the image temporarily
        image_path = os.path.join(app.root_path, 'static/images', imagefile.filename)
        imagefile.save(image_path)

        try:
            # Preprocess the image (e.g., convert to numpy array and scale)
            X = preprocess(image_path)
            encode_label = {0: "Benign", 1: "Malignant", 2: "Normal"}

            # Extract features
            features = feature_extractor.predict(X)
            features = features.reshape(features.shape[0], -1)

            # Predict using Random Forest Classifier
            Y_pred = classification_model.predict(features)
            probability = np.max(classification_model.predict_proba(features), axis=1) * 100
            Y_pred_index = int(Y_pred[0])
            probability_value = float(probability[0])
            classification = '%s (%.2f%%)' % (encode_label[Y_pred_index], probability_value)

            # Generate confidence bar chart
            probabilities = classification_model.predict_proba(features)[0] * 100
            classes = list(encode_label.values())

            # Set backend to avoid GUI issues
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            bars = plt.bar(classes, probabilities, color=['blue', 'red', 'green'])
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
            plt.ylim(0, max(probabilities) + 10)
            plt.xlabel('Classes')
            plt.ylabel('Confidence Level (%)')
            plt.title('Confidence Level for Each Class')

            # Save the plot temporarily
            chart_path = os.path.join(app.root_path, 'static', 'images', 'chart.png')
            plt.savefig(chart_path)
            plt.close()

            # Render the template with prediction and chart
            response = render_template('index.html', prediction=classification, image=imagefile.filename, chart=f'images/chart.png')

            # Schedule file deletion after 10 seconds
            files_to_delete = [image_path, chart_path]
            threading.Thread(target=delete_files_after_delay, args=(files_to_delete, 10)).start()

            return response

        except Exception as e:
            # Cleanup immediately in case of an error
            if os.path.exists(image_path):
                os.remove(image_path)
            return render_template('index.html', prediction=f"Error: {str(e)}", image=None)


if __name__ == '__main__':

    app.run(debug=True)
