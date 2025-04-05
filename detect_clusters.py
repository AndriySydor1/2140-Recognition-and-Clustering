import networkx as nx

# Припустимо, що граф G вже побудований
components = list(nx.connected_components(G))
cluster_assignment = {}
for cluster_id, comp in enumerate(components):
    for node in comp:
        cluster_assignment[node] = cluster_id
