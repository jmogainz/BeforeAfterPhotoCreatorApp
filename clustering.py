import os
import numpy as np
from tensorflow import keras
from collections import defaultdict

# plotting
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

ResNet152 = keras.applications.ResNet152
preprocess_input = keras.applications.resnet50.preprocess_input
image = keras.preprocessing.image
Model = keras.models.Model

# layer_name = 'conv5_block3_out'  # Example layer name, choose as per your need
# model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

from image_merging import get_image_timestamp
import datetime

def filter_out_non_before_after_images(clusters, input_dir):
    # if the difference between the creation dates of the first and last images in a cluster is less than 20 minutes, remove the cluster
    filtered_clusters = {}
    for cluster, filenames in clusters.items():
        timestamps = []
        for filename in filenames:
            timestamp = get_image_timestamp(os.path.join(input_dir, filename))
            if timestamp:
                timestamps.append(timestamp)
        if timestamps:
            timestamps.sort()
            if timestamps[1] - timestamps[0] > datetime.timedelta(minutes=5):
                filtered_clusters[cluster] = filenames
                
    return filtered_clusters

def load_images(image_directory):   
    image_list = []
    filenames = []
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG") or filename.endswith(".jpeg"):
            img_path = os.path.join(image_directory, filename)
            img = image.load_img(img_path, target_size=(512, 512))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            image_list.append(img_array)
            filenames.append(filename)
        else:
            print(f"Skipping {filename} as it is not a valid image file")
    return np.vstack(image_list), filenames

from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

def remove_duplicate_clusters(clusters):
    unique_clusters = {}
    seen_features = set()

    for cluster_key, features in clusters.items():
        # Sort the features for consistency in comparison
        sorted_features = tuple(sorted([filename for filename, _ in features]))
        
        if sorted_features not in seen_features:
            seen_features.add(sorted_features)
            unique_clusters[cluster_key] = features

    return unique_clusters

def cluster_images(image_directory):
    # Load images
    images, filenames = load_images(image_directory)
    
    # Load the pre-trained model
    model = ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    
    # Extract features
    feature_vectors = model.predict(images)
    
    # plot the feature vectors
    # plot_feature_vectors(feature_vectors, labels=filenames)
    
    # Reshape the features so that each image is represented by a single vector
    feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)

    def calculate_distance(feature1, feature2):
        return np.linalg.norm(feature1 - feature2)

    # Calculate distances between all feature vectors
    distances = {}
    for i, feature1 in enumerate(feature_vectors):
        distances[i] = []
        for j, feature2 in enumerate(feature_vectors):
            if i != j:
                dist = calculate_distance(feature1, feature2)
                distances[i].append((dist, j))
        # Sort distances for each feature vector
        distances[i].sort(key=lambda x: x[0])

    # calculate the average factor between the first and second closest matches
    average_closest_to_2nd_closest_factor = 0
    for i, dist_list in distances.items():
        average_closest_to_2nd_closest_factor += dist_list[1][0] / dist_list[0][0]
    average_closest_to_2nd_closest_factor /= len(distances)

    # Perform custom clustering
    clusters = []

    # Form clusters
    for i, dist_list in distances.items():
        # Always cluster with the closest
        closest = dist_list[0][1]
        clusters.append((i, closest))

        if dist_list[1][0] / dist_list[0][0] <= .5 * (average_closest_to_2nd_closest_factor - 1) + 1:
            clusters.append((i, dist_list[1][1]))
        elif dist_list[1][0] / dist_list[0][0] <= average_closest_to_2nd_closest_factor:
            for dist, j in dist_list[1:4]:
                clusters.append((i, j))

    # Remove duplicate clusters regardless of order inside the tuple
    clusters = list(set([tuple(sorted(pair)) for pair in clusters]))

    # Map clusters back to filenames
    clustered_filenames = {k: (filenames[pair[0]], filenames[pair[1]]) for k, pair in enumerate(clusters)}

    # Remove clusters that have close timestamps
    clustered_filenames = filter_out_non_before_after_images(clustered_filenames, image_directory)

    return clustered_filenames