import numpy as np


class AgglomerativeClustering:
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.clusters = []
    
    def fit(self, X):
        self.X = X
        clusters = [[x] for x in range(len(X))]
        
        while len(clusters) > self.n_clusters:
            nearest_clusters = self.find_nearest_clusters(clusters)
            new_cluster = clusters[nearest_clusters[0]] + clusters[nearest_clusters[1]]
            clusters.append(new_cluster)
            del clusters[nearest_clusters[1]]
            del clusters[nearest_clusters[0]]
        self.clusters = clusters
    
    def cluster_distance(self, cluster1, cluster2):
        sum_distance = 0
        for data_point_index in cluster1:
            sum_distance += np.sum(np.linalg.norm(self.X[data_point_index] - self.X[cluster2], axis=1))
        return sum_distance / (len(cluster1) * len(cluster2))
    
    def find_nearest_clusters(self, clusters):
        nearest_clusters = None
        min_distance = None
        for index1, cluster1 in enumerate(clusters):
            for index2, cluster2 in enumerate(clusters):
                if index2 <= index1:
                    continue
                distance = self.cluster_distance(cluster1, cluster2)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    nearest_clusters = [index1, index2]
        return nearest_clusters
