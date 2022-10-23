import numpy as np


class KNearestNeighbor:
    
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        dists = self.compute_distances(x_test)
        return self.predict_labels(dists)
    
    @staticmethod
    def distance(train, test):
        return (train - test) ** 2
    
    def compute_distances(self, x_test):
        dists = np.array(
            [
                self.distance(test, train)
                for test in x_test
                for train in self.X_train
            ]
        ).reshape((x_test.shape[0], self.X_train.shape[0]))
        
        return dists
    
    def predict_labels(self, dists):
        neighbours = np.empty((dists.shape[0], self.k), dtype='int32')
        y_pred = np.empty(dists.shape[0])
        for i in range(dists.shape[0]):
            neighbours[i] = np.argsort(dists[i])[:self.k]
            y_pred[i] = self.y_train[neighbours[i]].mean()
        
        return y_pred
    
    def score(self, x, y):
        var_result = ((y - self.predict(x)) ** 2).sum()
        var_total = ((y - y.mean()) ** 2).sum()
        
        return 1 - var_result / var_total
