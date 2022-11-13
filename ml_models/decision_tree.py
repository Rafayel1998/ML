import numpy as np


class DecisionNode:
    
    def __init__(
        self,
        feature_id=None,
        threshold=None,
        value=None,
        true_branch=None,
        false_branch=None
    ):
        self.feature_id = feature_id
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTreeClassifier:
    
    def __init__(
        self, impurity="gini",
        min_samples_split=2,
        max_depth=float("inf")
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        impurity_dict = {
            'gini': self.gini,
            'entropy': self.entropy
        }
        self.impurity = impurity_dict[impurity]
        self.labels = None
        self.root = None
    
    def gini(self, class_arr):
        p = np.bincount(class_arr, minlength=self.labels.shape[0]) / class_arr.shape[0]
        return 1 - (p ** 2).sum()
    
    def entropy(self, class_arr):
        p = np.bincount(class_arr, minlength=self.labels.shape[0]) / class_arr.shape[0]
        log_p = np.log2(p)
        log_p[log_p == -np.inf] = 0
        return - (p * log_p).sum()
    
    def purity_gain(self, y, y1, y2):
        return self.impurity(y) - \
               (y1.shape[0] * self.impurity(y1) +
                y2.shape[0] * self.impurity(y2)) / y.shape[0]
    
    @staticmethod
    def divide(x, y, feature_id, threshold):
        true_idx = x[:, feature_id] <= threshold
        return x[true_idx], y[true_idx], x[~true_idx], y[~true_idx]
    
    def grow_tree(self, x, y, current_depth=0):
        largest_purity_gain = 0
        nr_samples, nr_features = x.shape
        
        if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_id in range(nr_features):
                unique_values = np.unique(x[:, feature_id])
                for threshold in unique_values:
                    x1, y1, x2, y2 = self.divide(x, y, feature_id, threshold)
                    if len(x1) > 0 and len(x2) > 0:
                        purity_gain = self.purity_gain(y, y1, y2)
                        if purity_gain > largest_purity_gain:
                            largest_purity_gain = purity_gain
                            best_feature_id = feature_id
                            best_threshold = threshold
                            best_x1 = x1
                            best_y1 = y1
                            best_x2 = x2
                            best_y2 = y2
            
            if largest_purity_gain > 0:
                true_branch = self.grow_tree(
                    best_x1,
                    best_y1,
                    current_depth + 1
                )
                false_branch = self.grow_tree(
                    best_x2,
                    best_y2,
                    current_depth + 1
                )
                return DecisionNode(
                    feature_id=best_feature_id,
                    threshold=best_threshold,
                    true_branch=true_branch,
                    false_branch=false_branch
                )
        
        leaf_value = np.bincount(y, minlength=self.labels.shape[0])
        return DecisionNode(value=leaf_value)
    
    def fit(self, x_train, y_train):
        self.labels = np.unique(y_train)
        self.root = self.grow_tree(x_train, y_train)
    
    def get_leaf(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_id]
        branch = tree.true_branch if feature_value <= tree.threshold else tree.false_branch
        return self.get_leaf(x, branch)
    
    def predict(self, x_test):
        y_pred = [self.labels[np.argmax(self.get_leaf(datapoint))]
                  for datapoint in x_test]
        return np.array(y_pred)
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.where(y_pred == y_test)[0].shape[0] / y_test.shape[0]
