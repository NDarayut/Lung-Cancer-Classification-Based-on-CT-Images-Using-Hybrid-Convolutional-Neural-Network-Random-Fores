Deployed at [Hugging Face Space](https://huggingface.co/spaces/Darayut/Lung-Cancer-Prediction)  
Dataset can be found on [kaggle](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)   
Notebook: [Google Colab](https://drive.google.com/file/d/1qzl51JJZ19glgs6r2TXPL3mOHNOHfodG/view?usp=sharing)  
Dataset that were used in this research: [Google drive](https://drive.google.com/drive/folders/1RxW7MIiroNeksh-AmE8uLSxF6UbX9mLl?usp=sharing)  
Note: Only works with python 3.11

<h2>Abstract </h2>   
Lung cancer is a type of cancer that starts when abnormal cells grow in an uncontrolled way in the lungs. It is a serious health issue that can cause severe harm and death. Cancer that is caught at an early stage can be treated and could potentially saves lives. However, only 16% of lung cancer are diagnosed at an early stage, meaning more than 80% of lung cancer are diagnosed at a much later stage which could drastically reduce the survival rate of lung cancer patient. In this research, deep-learning and machine learning method is used to accurately identify the type of nodules within the lungs by using CT-images as input. CT-scan is one of the methods used to identify lung cancer, but radiologist struggle to identify the cancerous tumor residing in the lungs. With the help of technology and Artificial Intelligence, radiologist can use these tools to assist them in identifying the type of tumor and could further decreased the mortality rate of lung cancer. Through this research a dataset collected from the Iraqi hospitals was used on the hybrid convolutional neural network and random forest model (CNN-RF) to classify the type of nodule: benign, normal or malignant. The proposed model gives high accuracy ups to 94% on the testing set. The other performance metrices comes with high values such as 93% on recall average and 95% on precision average.   
<h2>About Dataset</h2>   
The Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) lung cancer dataset was collected in the above-mentioned specialist hospitals over a period of three months in fall 2019. It includes CT scans of patients diagnosed with lung cancer in different stages, as well as healthy subjects. IQ-OTH/NCCD slides were marked by oncologists and radiologists in these two centers. The dataset contains a total of 1190 images representing CT scan slices of 110 cases. These cases are grouped into three classes: normal, benign, and malignant. of these, 40 cases are diagnosed as malignant; 15 cases diagnosed with benign; and 55 cases classified as normal cases. The CT scans were originally collected in DICOM format. The scanner used is SOMATOM from Siemens. CT protocol includes: 120 kV, slice thickness of 1 mm, with window width ranging from 350 to 1200 HU and window center from 50 to 600 were used for reading. with breath hold at full inspiration. All images were de-identified before performing analysis. Written consent was waived by the oversight review board. The study was approved by the institutional review board of participating medical centers. Each scan contains several slices. The number of these slices range from 80 to 200 slices, each of them represents an image of the human chest with different sides and angles. The 110 cases vary in gender, age, educational attainment, area of residence and living status. Some of them are employees of the Iraqi ministries of Transport and Oil, others are farmers and gainers. Most of them come from places in the middle region of Iraq, particularly, the provinces of Baghdad, Wasit, Diyala, Salahuddin, and Babylon.   
<h2>Convolutional Neural Network</h2>  
Convolutional Neural Networks are a class of deep learning models particularly effective for image analysis. They consist of multiple layers, including convolutional layers, pooling layers, and fully connected layers, enabling them to automatically learn and extract features from images.  
<h3>Key Features</h3>  

- **Feature Extraction**: CNNs learn to identify relevant features from images, such as edges, textures, and shapes.  
- **Spatial Hierarchy**: By stacking multiple layers, CNNs can learn hierarchical representations, making them effective for complex image classification tasks.  
<h3>Implementation</h3>  
The CNN in this model is trained on medical images to extract relevant features that indicate the presence of malignant nodule. The CNN only act as a feature extractor and not a classifier.  
<h2>Random Forest</h2>  
Random Forest is an ensemble learning method that combines multiple decision trees to improve classification accuracy and robustness. It uses a bagging technique called bootstrap aggregating and feature bagging to train each tree on a random subset of the data as well as features, enhancing generalization and reducing overfitting.  
<h3>Key Features</h3>  

- **Ensemble Learning**: Random Forest aggregates the predictions of multiple decision trees, leading to a more accurate and stable model than individual trees.  
- **Robustness to Noise**: The ensemble approach makes Random Forest resilient to outliers and noise in the dataset, improving performance on real-world data.  
- **Bootstrap Aggregating (Bagging)**: Each tree in the forest is trained on a bootstrap sample (random sampling with replacement) of the dataset, which helps to reduce variance and improve the overall model performance.
- **Performance with Small Datasets**: Random Forest can perform well even with smaller datasets due to its inherent ability to generalize from limited samples. The aggregation of multiple trees helps to mitigate overfitting, making it a viable option when data is scarce.
<h3>Implementation</h3>  
In this model, Random Forest is used to classify lung cancer based on the features extracted by the CNN.  
<h2>CNN-RF Model</h2>  
Combining CNNs and Random Forests leverages the strengths of both models, creating a powerful hybrid approach:  

- **CNN for Feature Extraction**: The CNN automatically learns and extracts high-level features from medical images, which are often complex and challenging to interpret.  
- **Random Forest for Classification**: The Random Forest classifier utilizes the features extracted by the CNN to make final classifications, improving accuracy and robustness.
<h3>Benefits of the Hybrid Approach</h3>  

- **Improved Accuracy**: The combination of deep learning and ensemble methods can enhance overall classification performance.  
- **Robustness**: The hybrid model benefits from the strengths of both approaches, offering resilience to noise and outliers while effectively learning from limited data.  
<h2>Model Architecture</h2>  

![cnn-rf architecture](CNN-RF-Schematic.png)

<h2>How to setup</h2>  

1. Clone this repository:  
```git clone https://github.com/NDarayut/Lung-Cancer-Classification-With-CNN-RF-Model.git```  
2. Install virtual environment:  
<code>pip install virtualenv</code>
3. Create virtual environment:  
<code>python(version) -m venv (virtual-environment-name)</code>  
4. Activate virtual environment:  
<code>.\virtual-environment-name\Scripts\Activate</code>  
5. Install packages:  
<code>pip install -r requirements.txt </code>  
6. Run program:  
<code>python app.py</code>  
7. Get address and paste in browser:  
```http://127.0.0.1:5000```  
8. Deactivate environment:  
<code>deactivate</code>

<h2>Demo</h2>  

![Gif demo 3](https://github.com/user-attachments/assets/919769c6-7ae3-488a-9a29-109a0519d052)

<h2>References</h2>   
(alyasriy, hamdalla; AL-Huseiny, Muayed (2023), “The IQ-OTH/NCCD lung cancer dataset”, Mendeley Data, V4, doi: 10.17632/bhmdr45bh2.4)   
