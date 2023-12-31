import os
import numpy as np
from tensorflow import keras
from collections import defaultdict
from PIL import Image

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

from image_merging import get_image_timestamp, apply_exif_orientation
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

def load_images(image_directory, fix_orientation=True):
    image_list = []
    filenames = []
    rotated = False

    for filename in os.listdir(image_directory):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".JPEG")):
            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path)
            if fix_orientation:
                img, rotated = apply_exif_orientation(img)
            img = img.resize((512, 512))
            
            img_array = np.array(img)
            if img_array.ndim == 2 or img_array.shape[2] == 1:
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            image_list.append(img_array)
            filenames.append(filename)
        else:
            print(f"Skipping {filename} as it is not a valid image file")
    
    return np.vstack(image_list), filenames, rotated

from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

def calculate_pairs(model, images, filenames):
    feature_vectors = model.predict(images)

    if len(feature_vectors) == 2:
        return {0: (filenames[0], filenames[1])}
    
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

        # add logging to log the top 3 distances for each image
        # print(f"Closest 5 distances for {filenames[i]}: {dist_list[0][0]}: {filenames[dist_list[0][1]]}, {dist_list[1][0]}: {filenames[dist_list[1][1]]}, {dist_list[2][0]}: {filenames[dist_list[2][1]]}, {dist_list[3][0]}: {filenames[dist_list[3][1]]}, {dist_list[4][0]}: {filenames[dist_list[4][1]]}")
        # print(f"Factor between the first and second closest matches: {dist_list[1][0] / dist_list[0][0]}")
        # print(f"Average factor between the first and second closest matches: {average_closest_to_2nd_closest_factor}")

        try:
            number_of_contingency_clusters = len(dist_list) // 3
            if number_of_contingency_clusters < 2:
                number_of_contingency_clusters = 2 # minimum of 2 contingency clusters
            if dist_list[1][0] / dist_list[0][0] <= average_closest_to_2nd_closest_factor:
                for dist, j in dist_list[1:number_of_contingency_clusters]:
                    clusters.append((i, j))
        except IndexError:
            pass

    # Remove duplicate clusters regardless of order inside the tuple
    clusters = list(set([tuple(sorted(pair)) for pair in clusters]))

    return clusters

def cluster_images(image_directory):

    model = ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))

    # Load images
    images, filenames, rotated = load_images(image_directory, True)

    print(f"Rotated: {rotated}")

    clusters = calculate_pairs(model, images, filenames)

    # filter out clusters with timestamps that are too close together
    clustered_filenames = {k: (filenames[pair[0]], filenames[pair[1]]) for k, pair in enumerate(clusters)}

    clustered_filenames = filter_out_non_before_after_images(clustered_filenames, image_directory)

    return clustered_filenames