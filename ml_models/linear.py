import numpy as np
from sklearn.datasets import make_regression

make_reg = make_regression


class LinearRegression:
    
    def __init__(self, lr=0.1, tol=0.1, max_iter=1e5, adapt=True):
        self.lr = lr
        self.w0 = np.random.uniform() - 0.5
        self.w1 = np.random.uniform() - 0.5
        self.tol = tol
        self.max_iter = max_iter
        self.adapt = adapt
    
    def cost_function_grad(self, x, y):
        return -2 * (y - self.w0 - self.w1 * x)
    
    def update_weights(self, x, y):
        w0_step = self.lr * self.cost_function_grad(x, y)
        self.w0 -= w0_step
        self.w1 -= w0_step * x
    
    def fit(self, x_train, y_train):
        i = 0
        while True:
            for x, y in zip(x_train, y_train):
                old_w0 = self.w0
                old_w1 = self.w1
                self.update_weights(x, y)
                change = abs(self.w0 - old_w0) + abs(self.w1 - old_w1)
                if change < 2 * self.tol:
                    print("iter #: ", i)
                    print("change: ", change)
                    if change == 0:
                        print('gradient stopped changing')
                        return
                    if self.adapt:
                        self.lr /= 10
                        self.tol /= 10
                    else:
                        return
                if i == self.max_iter:
                    print('hit max iter')
                    return
                i += 1
    
    def predict(self, x_test):
        return self.w0 + self.w1 * x_test


class RidgeRegression:
    
    def __init__(self, lmb=0.1, lr=0.1, tol=0.1, max_iter=1e4, adapt=True):
        self.lmb = lmb
        self.lr = lr
        self.w = None
        self.tol = tol
        self.max_iter = max_iter
        self.adapt = adapt
        self.zero_ones = [0]
    
    def cost_function_grad(self, x, y):
        return -2 * (y - (x @ self.w))[:, np.newaxis] * x + 2 * self.lmb * self.w * self.zero_ones
    
    def update_weights(self, x, y):
        self.w -= self.lr * self.cost_function_grad(x, y).sum(axis=0) / x.shape[0]
    
    def fit(self, x_train, y_train):
        self.zero_ones += [1] * x_train.shape[1]
        self.zero_ones = np.array(self.zero_ones)
        x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
        self.w = np.random.uniform(size=x_train.shape[1]) - 0.5
        i = 0
        while True:
            old_w = np.array(self.w)
            self.update_weights(x_train, y_train)
            change = abs(self.w - old_w).sum()
            if change < x_train.shape[1] * self.tol:
                print("iter #: ", i)
                print("change: ", change)
                if change == 0:
                    print('gradient stopped changing')
                    return
                if self.adapt:
                    self.lr /= 10
                    self.tol /= 10
                else:
                    return
            if i == self.max_iter:
                print('hit max iter')
                return
            i += 1
    
    def predict(self, x_test):
        return np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1) @ self.w


class LogisticRegression:
    
    def __init__(self, lr=0.1, tol=0.01, max_iter=1e4, adapt=True):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.adapt = adapt
        self.w = None
    
    def cost_function_grad(self, x, y):
        return (self.predict_proba(x[:, 1:]) - y) @ x / x.shape[0]
    
    def update_weights(self, x, y):
        self.w -= self.lr * self.cost_function_grad(x, y)
    
    def fit(self, x_train, y_train):
        x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
        self.w = np.random.uniform(size=x_train.shape[1]) - 0.5
        i = 0
        while True:
            old_w = np.array(self.w)
            self.update_weights(x_train, y_train)
            change = abs(self.w - old_w).sum()
            if change < self.tol:
                print("iter #: ", i)
                print("change: ", change)
                if change == 0:
                    print('gradient stopped changing')
                    return
                if self.adapt:
                    self.lr /= 10
                    self.tol /= 10
                else:
                    return
            if i == self.max_iter:
                print('hit max iter')
                return
            i += 1
    
    def predict(self, x_test, threshold=0.5):
        return (1 - np.signbit(self.predict_proba(x_test) - threshold)) * 1
    
    def predict_proba(self, x_test):
        return self.sigmoid(np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=-1) @ self.w)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
