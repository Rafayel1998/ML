import numpy as np


class DiscriminantAnalysis:
    
    def __init__(self, model):
        assert model in ['Linear', 'Quadratic']
        self.model = model
        self.labels = None
        self.means = {}
        self.covariance_matrices = {}
        self.covariance_matrix_inverse = None
        self.label_log_probabilities = {}
    
    @staticmethod
    def get_stats(x):
        mean = x.mean(axis=0)
        covariance_matrix = np.cov(x, ddof=x.shape[1], rowvar=False)
        return mean, covariance_matrix
    
    @staticmethod
    def get_label_log_probabilities(y):
        return np.log(np.bincount(y) / y.shape[0])
    
    def ldf(self, x, mean, covariance_matrix_inverse):
        if self.model == 'Linear':
            product = covariance_matrix_inverse @ mean
            return x @ product - mean.T @ product / 2
        else:
            return -0.5 * (-np.log(np.linalg.det(covariance_matrix_inverse)) + np.diag(
                ((x - mean) @ covariance_matrix_inverse @ (x - mean).T)
            ))
    
    def fit(self, x_train, y_train):
        self.labels = np.unique(y_train)
        label_log_probabilities = self.get_label_log_probabilities(y_train)
        self.label_log_probabilities = dict(zip(self.labels, label_log_probabilities))
        for label in self.labels:
            idx = np.where(y_train == label)
            mean, covariance_matrix = self.get_stats(x_train[idx])
            self.means[label] = mean
            self.covariance_matrices[label] = covariance_matrix
        if self.model == 'Linear':
            self.covariance_matrix_inverse = np.linalg.inv(
                np.mean([*self.covariance_matrices.items()])
            )
    
    def predict(self, x_test):
        return self.labels[self.predict_probabilities(x_test).argmax(axis=0)]
    
    def predict_probabilities(self, x_test):
        probabilities = []
        for label in self.labels:
            mean = self.means[label]
            if self.model == 'Linear':
                covariance_matrix_inverse = self.covariance_matrix_inverse
            else:
                covariance_matrix_inverse = np.linalg.pinv(self.covariance_matrices[label])
            prob = self.label_log_probabilities[label] + self.ldf(x_test, mean, covariance_matrix_inverse)
            probabilities.append(prob)
        return np.array(probabilities)
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.where(y_pred == y_test)[0].shape[0] / y_test.shape[0]
