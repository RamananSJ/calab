import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def assign_data_to_clusters(data, centroids):
    clusters = {centroid_name: [] for centroid_name in centroids.keys()}

    for point in data:
        closest_centroid = min(centroids.keys(), key=lambda c: euclidean_distance(point, centroids[c]))
        clusters[closest_centroid].append(point)

    return clusters

def update_centroids(clusters):
    new_centroids = {}

    for centroid_name, cluster_points in clusters.items():
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids[centroid_name] = new_centroid.tolist()
        else:
            new_centroids[centroid_name] = centroids[centroid_name]  # If a cluster is empty, keep the old centroid

    return new_centroids

# Initialize data and centroids
data = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]
centroids = {"centroid1": [2,10], "centroid2": [5,8], "centroid3": [1,2]}

# K-means clustering
num_iterations = 5  # You can adjust the number of iterations
for iteration in range(num_iterations):
    clusters = assign_data_to_clusters(data, centroids)
    new_centroids = update_centroids(clusters)

    # Print the clusters and centroids at each iteration
    print(f"Iteration {iteration + 1}:")
    for centroid_name, cluster_points in clusters.items():
        print(f"{centroid_name}: {cluster_points}")

    centroids = new_centroids

# Final clusters and centroids
print("Final Clusters:")
for centroid_name, cluster_points in clusters.items():
    print(f"{centroid_name}: {cluster_points}")

print("Final Centroids:")
for centroid_name, centroid_location in centroids.items():
    print(f"{centroid_name}: {centroid_location}")
