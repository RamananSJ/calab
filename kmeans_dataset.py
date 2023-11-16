import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

# Load the dataset
data = pd.read_csv('data4_19.csv')

# Extract the features (first four columns)
X = data.iloc[:, :4].values

# Initialize K-means with K=3 clusters and random initialization
kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=42)

# Fit the model
kmeans.fit(X)

# Get the cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Get the ground truth cluster labels from the last column
ground_truth_labels = data.iloc[:, -1].values

# Print the final cluster centers
print("Final Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1}: {center}")

# Compute Jaccard distances for each cluster
jaccard_distances = []
for i in range(3):
    ground_truth_cluster = np.where(ground_truth_labels == f'Iris {i + 1}')[0]
    predicted_cluster = np.where(cluster_labels == i)[0]

    # Compute Jaccard distance using the intersection over union
    intersection = len(set(ground_truth_cluster).intersection(predicted_cluster))

    union = len(set(ground_truth_cluster).union(predicted_cluster))
 
    jaccard_distance = 1 - (intersection / union)

    jaccard_distances.append(jaccard_distance)

# Print the Jaccard distances for each cluster
print("\nJaccard Distances:")
for i, distance in enumerate(jaccard_distances):
    print(f"Cluster {i + 1}: {distance:.4f}")
