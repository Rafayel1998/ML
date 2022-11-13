import numpy as np


class KMeans:
    def __init__(self, k, method='random', max_iter=300):
        self.k = k
        self.method = method
        self.max_iter = max_iter
        self.centroids = None
        self.clusters = None

    def init_centers(self, X):
        if self.method == 'random':
            return X[np.random.choice(X.shape[0], self.k, replace=False)]
        if self.method == 'k-means++':
            centroids = []
            weights = np.ones(X.shape[0]) / X.shape[0]
            for i in range(self.k):
                centroids.append(X[np.random.choice(X.shape[0], 1, p=weights)][0])
                if i == self.k:
                    break
                weights = np.sqrt(
                    np.square(
                        np.array(centroids)[:, np.newaxis] - X
                    ).sum(axis=-1)
                ).min(axis=0)
                weights = weights / weights.sum()
            return np.array(centroids)

    def fit(self, X):
        self.centroids = self.init_centers(X)
        for _ in range(self.max_iter):
            self.clusters = self.expectation(X)
            new_centroids = self.maximization(X)
            if (new_centroids == self.centroids).all():
                break
            self.centroids = new_centroids

    def expectation(self, X):
        return np.sqrt(
            np.square(
                self.centroids[:, np.newaxis] - X
            ).sum(axis=-1)
        ).argmin(axis=0)

    def maximization(self, X):
        new_centroids = []
        for i in range(self.k):
            new_centroids.append(X[self.clusters == i].mean(axis=0))
        new_centroids = np.array(new_centroids)
        return new_centroids

    def predict(self, X):
        return self.expectation(X)

    def predict_proba(self, X):
        dist_matrix = np.sqrt(
            np.square(
                self.centroids[:, np.newaxis] - X
            ).sum(axis=-1)
        )
        return (dist_matrix / dist_matrix.sum(axis=0)).T