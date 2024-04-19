#main code
'''Isolation Forest is a powerful algorithm commonly used for detecting anomalies, both logical and 
structural, in unsupervised data. It works by isolating anomalies in a dataset using a tree-based approach.'''
import numpy as np  
import pandas as pd  
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Directory containing the images
image_folder = 'train/good/'  # Replace 'path/to/image/folder' with the actual folder path

# Get all image file names from the folder
image_files = os.listdir(image_folder)

# Create full image paths by joining the folder path with each image file name
image_paths = [os.path.join(image_folder, img_file) for img_file in image_files]


#ignores if any warning arises to continue running of code
warnings.filterwarnings("ignore")

# Function to extract features from an image using color histogram
def extract_features(image_path, bins=(8, 8, 8)):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to a fixed size (optional)
    image = cv2.resize(image, (256, 256))
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    # Flatten the histogram to create a feature vector for each individual image data point
    features = hist.flatten()
    return features

feature_list = [extract_features(img_path) for img_path in image_paths]
feature_matrix = np.array(feature_list)



# Initialize and fit Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1)  # Adjust contamination based on expected anomalies
isolation_forest.fit(feature_matrix)

# Predict anomalies
anomaly_scores = isolation_forest.decision_function(feature_matrix)
anomaly_labels = isolation_forest.predict(feature_matrix)


# Identify anomalies based on scores and labels
# Calculate threshold based on training data
threshold = np.percentile(anomaly_scores, 100)  # Since all values are good we set it 100 
print(threshold)



# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(anomaly_scores, bins=50)
plt.title('Anomaly Scores Histogram')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=anomaly_labels, cmap='viridis')
plt.colorbar(label='Anomaly Label')
plt.title('Anomaly Scores vs. Index')
plt.xlabel('Index')
plt.ylabel('Anomaly Score')

plt.tight_layout()
plt.show()

test_folder = 'test'
# List to store all image file paths
test_image_paths = []


test_labels=[]
for folder in os.listdir(test_folder):
    test_labels.append(folder)

for i,folder in enumerate(test_labels):
    for image in os.listdir(test_folder+'/'+folder):
        img_path = os.path.join(test_folder+'/'+folder+'/'+image)
        test_image_paths.append(img_path)

test_feature_list = [extract_features(img_path) for img_path in test_image_paths]
test_feature_matrix = np.array(test_feature_list)

# Predict anomalies
test_anomaly_scores = isolation_forest.decision_function(test_feature_matrix)
test_anomaly_labels = isolation_forest.predict(test_feature_matrix)

logical_anomalies=[]
structural_anomalies=[]

print(len(test_anomaly_labels), len(test_image_paths))

# Iterate through test anomaly labels and scores
for i, label in enumerate(test_anomaly_labels):
    print(f"Index: {i}, Label: {label}")
    if label == -1:  # Check index validity
        logical_anomalies.append(test_image_paths[i])

# Iterate through test anomaly scores for structural anomalies
for i, score in enumerate(test_anomaly_scores):
    print(f"Index: {i}, Score: {score}")
    if score >= threshold:  # Check index validity for structural anomalies
        structural_anomalies.append(test_image_paths[i])

# Print and visualize anomalies
print("\nThese images from test dataset show Structural Anomalies:")
for img_path in structural_anomalies:
    print(img_path)


print("These images from test dataset show Logical Anomalies:")
for img_path in logical_anomalies:
    print(img_path)
