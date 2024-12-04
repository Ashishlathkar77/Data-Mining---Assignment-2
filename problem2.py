import numpy as np
import random

# Function to compute the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-means algorithm
def k_means(X, k, max_iter=100, tolerance=0.001, random_init=10):
    best_sse = float('inf')
    best_centroids = None
    
    for _ in range(random_init):
        # Randomly initialize centroids
        centroids = X[random.sample(range(X.shape[0]), k)]
        
        prev_sse = float('inf')
        
        for i in range(max_iter):
            # Assignment step: assign each point to the closest centroid
            clusters = [[] for _ in range(k)]
            for point in X:
                distances = [euclidean_distance(point, centroid) for centroid in centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(point)
            
            new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0])
            
            # Compute SSE (Sum of Squared Errors)
            sse = 0
            for cluster_idx, cluster in enumerate(clusters):
                for point in cluster:
                    sse += euclidean_distance(point, centroids[cluster_idx]) ** 2
            
            # Convergence check: if the change in SSE is small enough or max iterations reached
            if abs(prev_sse - sse) < tolerance:
                break
            prev_sse = sse
            centroids = new_centroids
            
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            
    return best_sse

data_path = '/workspaces/Data-Mining---Assignment-2/Data_Problem2/seeds.txt' 
X = np.loadtxt(data_path)

# Run k-means clustering for k=3, k=5, k=7 and average over 10 initializations
k_values = [3, 5, 7]
for k in k_values:
    sse_values = []
    for _ in range(10):
        sse = k_means(X, k)
        sse_values.append(sse)
    
    avg_sse = np.mean(sse_values)
    print(f"Average SSE for k={k}: {avg_sse:.4f}")
