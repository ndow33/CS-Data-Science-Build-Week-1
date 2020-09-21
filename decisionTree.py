import numpy as np
import pandas as pd

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:

    # max depth determines how many layers the tree will have.
    # when instantiating, it needs at least one layer to run
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

#######################################################################################   
    def fit(self, X, y):
        # n_classes = the total possible number of outcomes
        self.n_classes = len(set(y)) # creates a set of the unique possible outcomes
        self.n_features = X.shape[1] # the number of features in the given data
        self.tree = self._grow_tree(X, y) # creates a decision tree from the data
        ### go to the _create_tree function...

#######################################################################################
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)] # what does this return?
        predicted_class = np.argmax(num_samples_per_class) # what does this return?
        node = Node(predicted_class=predicted_class)

        if depth < self.max_depth:
            idx, thr = self._best_split(X,y)
            ### go to the best_split function
            if idx is not None:
                indeces_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]


#######################################################################################
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
            ### return to _grow_tree

        num_parent = [np.sum(y == c) for c in range(self.n_classes)] # what does this return?
        best_gini = 1.0 - sum((n / m)**2 for n in num_parent) # what does this return?
        best_idx, best_thr = None, None

        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y))) # what does this return? it returns two lists: thresholds and classes
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1] # this uses the list classes from above to return a single class, or answer (a class is the y or target value)
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes)) # returns the gini impurity for the left fork using comprehension
                gini_right = 1.0 - sum((num_right[x] / (m - i))**2 for x in range(self.n_classes)) # returns the gini impurity for the right fork using comprehension
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                    
        return best_idx, best_thr
        ### return to _grow_tree
#######################################################################################
    def predict(self, features, target):
        # returns a list of predictions
        return [self._predict(inputs) for inputs in X]

