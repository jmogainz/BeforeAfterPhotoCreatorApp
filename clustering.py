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
    
    # Extract features
    # Load the pre-trained model
    model = ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    features = model.predict(images)
    
    # reshape the features so that each image is represented by a single vector
    features = features.reshape(features.shape[0], -1)

    num_clusters = len(filenames) // 2 + 1

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=22)
    kmeans.fit(features)

    # Initial clustering
    clusters = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append((filenames[i], features[i]))

    # Function to calculate distance between two features
    def calculate_distance(feature1, feature2):
        return np.linalg.norm(feature1 - feature2)

    re_cluster_count = 0
    new_cluster_count = 0

    for cluster, features in list(clusters.items()):
        if len(features) > 2:
            num_clusters = len(features) // 2
            if len(features) % 2 == 1:
                num_clusters += 1

            cluster_features = [feature for _, feature in features]
            kmeans = KMeans(n_clusters=num_clusters, random_state=22)
            kmeans.fit(cluster_features)

            temp_new_clusters = {}
            for i, label in enumerate(kmeans.labels_):
                new_cluster_key = f"re_cluster_{re_cluster_count}_{label}"
                if new_cluster_key not in temp_new_clusters:
                    temp_new_clusters[new_cluster_key] = []
                temp_new_clusters[new_cluster_key].append(features[i])

            del clusters[cluster]
            
            for new_cluster_key, new_cluster_features in temp_new_clusters.items():
                new_cluster_key_cluster_modified = False
                if len(new_cluster_features) == 1:
                    single_feature = new_cluster_features[0][1]
                    closest_match = None
                    closest_distance = float('inf')

                    # Find closest match
                    for other_cluster_key, other_cluster_features in temp_new_clusters.items():
                        if other_cluster_key != new_cluster_key:
                            for filename, feature in other_cluster_features:
                                distance = calculate_distance(single_feature, feature)
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_match = (filename, feature, other_cluster_key)

                    # Identify matches within 5% of the closest match's distance
                    matches_within_5_percent = []
                    for other_cluster_key, other_cluster_features in temp_new_clusters.items():
                        if other_cluster_key != new_cluster_key:
                            for filename, feature in other_cluster_features:
                                distance = calculate_distance(single_feature, feature)
                                if distance / closest_distance <= 1.05 and (filename, feature) != closest_match[:2]:
                                    matches_within_5_percent.append((filename, feature))

                    # Create new clusters or pair using cut method
                    if matches_within_5_percent:
                        for filename, feature in matches_within_5_percent:
                            new_5_percent_cluster_name = f"new_5_percent_re_cluster_{new_cluster_count}"
                            new_cluster_count += 1
                            clusters[new_5_percent_cluster_name] = [new_cluster_features[0], (filename, feature)]
                        new_5_percent_cluster_name = f"new_5_percent__re_cluster_{new_cluster_count}"
                        clusters[new_5_percent_cluster_name] = [new_cluster_features[0], closest_match[:2]]
                        new_cluster_count += 1
                    else:
                        if closest_match:
                            clusters[new_cluster_key] = [new_cluster_features[0], closest_match[:2]]
                            temp_new_clusters[closest_match[2]].remove(closest_match[:2])
                            new_cluster_key_cluster_modified = True

                if not new_cluster_key_cluster_modified:
                    clusters[new_cluster_key] = new_cluster_features

            re_cluster_count += 1

    single_feature_clusters = {k: v for k, v in clusters.items() if len(v) == 1}
    new_cluster_count = 0  # For 5% logic clusters

    for single_cluster_key, single_cluster_features in single_feature_clusters.items():
        try:
            single_feature = single_cluster_features[0][1]
        except IndexError as e:
            continue

         # Store all distances and corresponding features
        distances = []
        for cluster_key, features in clusters.items():
            if cluster_key != single_cluster_key:
                for filename, feature in features:
                    if filename != single_cluster_features[0][0]:
                        distance = calculate_distance(single_feature, feature)
                        # prevent duplicate additions to distances (only in considering the distance value)
                        if (distance) not in [d[0] for d in distances]:
                            distances.append((distance, (filename, feature, cluster_key)))

        # Sort the distances
        distances.sort(key=lambda x: x[0])

        # Get the closest, second closest, and third closest matches
        closest_match = distances[0][1] if len(distances) > 0 else (None, float('inf'))
        second_closest_match = distances[1][1] if len(distances) > 1 else (None, float('inf'))
        third_closest_match = distances[2][1] if len(distances) > 2 else (None, float('inf'))
        fourth_closest_match = distances[3][1] if len(distances) > 3 else (None, float('inf'))

        closest_distance = distances[0][0] if len(distances) > 0 else float('inf')

        # Identify matches within 5% of the closest match's distance
        matches_within_5_percent = []
        if closest_match:
            for cluster_key, features in clusters.items():
                if cluster_key != single_cluster_key:
                    for filename, feature in features:
                        if filename != single_cluster_features[0][0]:
                            distance = calculate_distance(single_feature, feature)
                            if distance / closest_distance <= 1.07 and (filename, feature) != closest_match[:2] and (filename, feature) not in matches_within_5_percent:
                                matches_within_5_percent.append((filename, feature))
                                
        # Create new clusters for matches within 5% or pair using cut method
        if matches_within_5_percent:
            for filename, feature in matches_within_5_percent:
                new_5_percent_cluster_name = f"new_5_percent_cluster_{new_cluster_count}"
                new_cluster_count += 1
                clusters[new_5_percent_cluster_name] = [single_cluster_features[0], (filename, feature)]
            new_5_percent_cluster_name = f"new_5_percent_cluster_{new_cluster_count}"
            clusters[new_5_percent_cluster_name] = [single_cluster_features[0], closest_match[:2]]
            new_cluster_count += 1
        elif closest_match:
            clusters[single_cluster_key] = [single_cluster_features[0], closest_match[:2]]

            if len(distances) > 1 and distances[1][0] / closest_distance > 1.08:
                new_cluster_name = f"new_additional_cluster_{new_cluster_count}"
                new_cluster_count += 1
                clusters[new_cluster_name] = [single_cluster_features[0], second_closest_match[:2]]
                new_cluster_name = f"new_additional_cluster_{new_cluster_count}"
                new_cluster_count += 1
                clusters[new_cluster_name] = [single_cluster_features[0], third_closest_match[:2]]
                new_cluster_name = f"new_additional_cluster_{new_cluster_count}"
                new_cluster_count += 1
                clusters[new_cluster_name] = [single_cluster_features[0], fourth_closest_match[:2]]
            else:
                clusters[closest_match[2]].remove(closest_match[:2])

    clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    clusters = remove_duplicate_clusters(clusters)

    clusters = {k: [filename for filename, _ in v] for k, v in clusters.items()}
    clusters = filter_out_non_before_after_images(clusters, image_directory)
    
    return clusters