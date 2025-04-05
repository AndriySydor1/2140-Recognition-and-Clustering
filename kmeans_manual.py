import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

k = 3
num_iterations = 100
n_samples = X_scaled.shape[0]

np.random.seed(42)
centroids = X_scaled[np.random.choice(n_samples, k, replace=False)]

for iteration in range(num_iterations):
    distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X_scaled])
    cluster_labels = np.argmin(distances, axis=1)
    new_centroids = np.array([X_scaled[cluster_labels == i].mean(axis=0) for i in range(k)])
    if np.allclose(centroids, new_centroids, atol=1e-4):
        break
    centroids = new_centroids
