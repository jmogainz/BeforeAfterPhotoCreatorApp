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

class TimestampError(Exception):
    """Exception raised when timestamps are missing in image metadata."""
    pass

log_level = "DEBUG"
def log(statement):
    if log_level == "DEBUG":
        print("[INFO] " + statement)

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
            else:
                raise TimestampError(f"Timestamps are missing in some or all image metadata.")
        if timestamps:
            timestamps.sort()
            if timestamps[1] - timestamps[0] > datetime.timedelta(minutes=5):
                filtered_clusters[cluster] = filenames
            else:
                log(f"Filtered cluster {cluster} with timestamps: {timestamps}")
                
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
        return [(0, 1)]
    
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

        # do the same logging statement above but do it in a for loop with a try except block to handle the case where there are less than 5 images in the cluster
        if log_level == "DEBUG":
            for j in range(5):
                try:
                    log(f"Closest {j+1} distances for {filenames[i]}: {dist_list[j][0]}: {filenames[dist_list[j][1]]}")
                except IndexError:
                    pass
            log(f"Factor between the first and second closest matches: {dist_list[1][0] / dist_list[0][0]}")
            log(f"Average factor between the first and second closest matches: {average_closest_to_2nd_closest_factor}")

        try:
            number_of_contingency_clusters = round(-8 * average_closest_to_2nd_closest_factor + 15.8) # trialed and error to get this formula
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

    log(f"Rotated: {rotated}")

    clusters = calculate_pairs(model, images, filenames)

    # filter out clusters with timestamps that are too close together
    clustered_filenames = {k: (filenames[pair[0]], filenames[pair[1]]) for k, pair in enumerate(clusters)}

    log(f"Number of clusters before timestamp filter: {len(clustered_filenames)}")
    clustered_filenames = filter_out_non_before_after_images(clustered_filenames, image_directory)
    log(f"Number of clusters after timestamp filter: {len(clustered_filenames)}")

    # print out clusters
    for cluster, filenames in clustered_filenames.items():
        log(f"Cluster {cluster}: {filenames}")

    return clustered_filenames