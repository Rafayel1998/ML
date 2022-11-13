import numpy as np
from scipy import stats
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, k=3, init='random', max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.covs = None
        self.proportions = None

    def init_params(self, X):
        proportions = np.array([1 / self.k] * self.k)
        n_cols = X.shape[1]
        mean_func = lambda x: np.mean(x, axis=0)
        cov_func = lambda x: np.cov(x, rowvar=False)
        if self.init == 'random':
            first_index = int((X.shape[0] + self.k - 1) / self.k)
            cluster_size = int(X.shape[0] / self.k)
            groups = np.split(np.random.permutation(X), range(first_index, X.shape[0], cluster_size))
            means = np.vstack(list(map(mean_func, groups)))
            covs = np.vstack(list(map(cov_func, groups))).reshape(self.k, n_cols, n_cols)
            return means, covs, proportions
        if self.init == 'kmeans':
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)
            args = np.argsort(y_kmeans)
            # cuts that will separate clusters
            cuts = np.where(y_kmeans[args][1:] - y_kmeans[args][:-1] == 1)[0] + 1
            groups = np.split(X[args], cuts)
            means = np.vstack(list(map(mean_func, groups)))
            covs = np.vstack(list(map(cov_func, groups))).reshape(self.k, n_cols, n_cols)
            return means, covs, proportions
        if self.init == 'random_means++':
            kmeans = KMeans(n_clusters=self.k, max_iter=1, tol=np.inf)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)
            args = np.argsort(y_kmeans)
            # cuts that will separate clusters
            cuts = np.where(y_kmeans[args][1:] - y_kmeans[args][:-1] == 1)[0] + 1
            groups = np.split(X[args], cuts)
            means = np.vstack(list(map(mean_func, groups)))
            covs = np.vstack(list(map(cov_func, groups))).reshape(self.k, n_cols, n_cols)
            return means, covs, proportions
        if self.init == 'random_means':
            means = X[np.random.choice(X.shape[0], self.k, replace=False)]
            covs = np.repeat(np.cov(X, rowvar=False)[np.newaxis, ...], self.k, axis=0)
            return means, covs, proportions
        if self.init == 'random_means_covariances':
            means = X[np.random.choice(X.shape[0], self.k, replace=False)]
            # generating random matrix
            covs = np.random.normal(size=(self.k, n_cols, n_cols))
            # multiplying matrices with their transpose to make it positive semidefinite
            covs_T = np.swapaxes(covs, 1, 2)
            covs = (covs[..., np.newaxis] * covs_T[:, np.newaxis, ...]).sum(axis=-2)
            return means, covs, proportions

    def fit(self, X):
        self.means, self.covs, self.proportions = self.init_params(X)
        for _ in range(self.max_iter):
            gamma = self.expectation(X)
            means, covs, self.proportions = self.maximization(X, gamma)
            if (means == self.means).all() and (covs == self.covs).all():
                break
            self.means = means
            self.covs = covs

    def expectation(self, X):
        gamma = []
        for mean, cov, proportion in zip(self.means, self.covs, self.proportions):
            dist = stats.multivariate_normal(mean, cov, allow_singular=True)
            gamma.append(proportion * dist.pdf(X))
        gamma = np.array(gamma)
        return (gamma / gamma.sum(axis=0)).T

    def maximization(self, X, gamma):
        n = gamma.sum(axis=0)
        means = (X[:, np.newaxis, :] * gamma[..., np.newaxis]).sum(axis=0) / n[:, np.newaxis]
        diff = X[:, np.newaxis, :] - means
        covs = (diff[..., np.newaxis] *
                diff[..., np.newaxis, :] *
                gamma[..., np.newaxis, np.newaxis]
                ).sum(axis=0) / n[:, np.newaxis, np.newaxis]
        proportions = n / n.sum()
        return means, covs, proportions

    def predict_proba(self, X):
        return self.expectation(X)
