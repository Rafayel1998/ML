import numpy as np
from scipy import stats


class NaiveBayes:

    def __init__(self, distribution='Gaussian'):
        self.distribution = distribution
        self.labels = None
        self.means = {}
        self.variances = {}
        self.minimums = {}
        self.maximums = {}
        self.label_log_probabilities = {}

    @staticmethod
    def get_stats(x):
        mean = x.mean(axis=0)
        variance = x.var(axis=0, ddof=x.shape[1])
        minimum = x.min(axis=0)
        maximum = x.max(axis=0)
        return mean, variance, minimum, maximum

    @staticmethod
    def get_label_log_probabilities(y):
        return np.log(np.bincount(y) / y.shape[0])

    def log_pdf(self, x, mean, variance, minimum, maximum):
        distributions = {
            'Gaussian': stats.norm(mean, variance),
            'Uniform': stats.uniform(minimum, maximum - minimum),
            'Exponential': stats.expon(scale=mean)
        }
        return distributions[self.distribution].logpdf(x)

    def fit(self, x_train, y_train):
        self.labels = np.unique(y_train)
        label_log_probabilities = self.get_label_log_probabilities(y_train)
        self.label_log_probabilities = dict(zip(self.labels, label_log_probabilities))
        for label in self.labels:
            idx = np.where(y_train == label)
            mean, variance, minimum, maximum = self.get_stats(x_train[idx])
            self.means[label] = mean
            self.variances[label] = variance
            self.minimums[label] = minimum
            self.maximums[label] = maximum

    def predict(self, x_test):
        return self.labels[self.predict_probabilities(x_test).argmax(axis=0)]

    def predict_probabilities(self, x_test):
        probabilities = []
        for label in self.labels:
            mean = self.means[label]
            variance = self.variances[label]
            minimum = self.minimums[label]
            maximum = self.maximums[label]
            prob = self.label_log_probabilities[label] + \
                self.log_pdf(x_test, mean, variance, minimum, maximum).sum(axis=1)
            probabilities.append(prob)
        return np.array(probabilities)

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.where(y_pred == y_test)[0].shape[0] / y_test.shape[0]
