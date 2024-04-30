import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# Load the pre-trained model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Assume images are stored in "/path/to/your/images"
image_dir = Path('/workspace/voiceai/genimage')
image_paths = list(image_dir.glob('*.png'))  # or any file extension your images have

# Preprocess and extract features for each image
features = []
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(1085, 305))  # Resize image
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)
    feature = model.predict(img_preprocessed)
    features.append(feature.flatten())

features = np.array(features)
# Reduce the features to 2 dimensions using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)


# Number of clusters/categories
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# `clusters` now contains the cluster assignment (0 to 4) for each image

for i in range(n_clusters):
    cluster_indices = np.where(clusters == i)[0]
    print(f"Images in cluster {i}:")
    for idx in cluster_indices:
        print(image_paths[idx])

# Create a scatter plot of the reduced features, coloring each point by its cluster label
plt.figure(figsize=(10, 6))

# Define a list of colors, one for each cluster
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if you have more than 7 clusters

for i in range(n_clusters):
    # Find points in this cluster
    ix = np.where(clusters == i)
    # Plot these points with the cluster's color
    plt.scatter(reduced_features[ix, 0], reduced_features[ix, 1], c=colors[i], label=f'Cluster {i}')

plt.title('Clustered Images')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
