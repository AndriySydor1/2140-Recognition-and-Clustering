import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

width, height = 7, 7
input_len = X.shape[1]
learning_rate = 0.5
sigma = 1.0
num_iterations = 100

weights = np.random.rand(width, height, input_len)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def decay(param, current_iter, max_iter):
    return param * np.exp(-current_iter / max_iter)

def get_bmu(x):
    min_dist = np.inf
    bmu_idx = (0, 0)
    for i in range(width):
        for j in range(height):
            dist = euclidean_distance(x, weights[i, j])
            if dist < min_dist:
                min_dist = dist
                bmu_idx = (i, j)
    return bmu_idx

for iteration in range(num_iterations):
    for x in X_scaled:
        bmu_i, bmu_j = get_bmu(x)
        lr = decay(learning_rate, iteration, num_iterations)
        sig = decay(sigma, iteration, num_iterations)
        for i in range(width):
            for j in range(height):
                dist_to_bmu = (i - bmu_i)**2 + (j - bmu_j)**2
                h = np.exp(-dist_to_bmu / (2 * (sig ** 2)))
                weights[i, j] += lr * h * (x - weights[i, j])
