import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

n = 7
np.random.seed(42)
points = np.random.rand(n, 2)
sigma = 0.3
W = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_sq = np.sum((points[i] - points[j]) ** 2)
        W[i, j] = np.exp(-dist_sq / (sigma ** 2))

threshold = 0.6
G = nx.Graph()
for i in range(n):
    G.add_node(i)
    for j in range(i + 1, n):
        if W[i, j] >= threshold:
            G.add_edge(i, j, weight=W[i, j])
