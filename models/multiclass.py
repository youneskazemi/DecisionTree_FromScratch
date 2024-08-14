import numpy as np
from itertools import combinations


class OneVsAllClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classifiers = []

    def fit(self, X, y, dtype_dict):
        self.classes_ = np.unique(y)
        self.classifiers = []
        for _cls in self.classes_:
            binary_y = (y == _cls).astype(int)
            classifier = self.base_estimator()
            classifier.fit(X, binary_y, dtype_dict)
            self.classifiers.append(classifier)

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for i, clf in enumerate(self.classifiers):
            class_probas = clf.predict_proba(X)[:, 1]
            
            probas[:, i] = class_probas
        return probas

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class OneVsOneClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.classifiers = {}
        self.pairs = []

    def fit(self, X, y, dtype_dict):
        self.classes_ = np.unique(y)
        self.pairs = list(combinations(self.classes_, 2))
        self.classifiers = {}

        for class1, class2 in self.pairs:
            indices = np.where((y == class1) | (y == class2))[0]
            X_pair = X[indices]
            y_pair = y[indices]
            y_pair = np.where(y_pair == class1, 0, 1)

            classifier = self.base_estimator()
            classifier.fit(X_pair, y_pair, dtype_dict)
            self.classifiers[(class1, class2)] = classifier

    def predict_proba(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))

        for (class1, class2), clf in self.classifiers.items():
            probabilities = clf.predict_proba(X)
            votes[:, class1] += probabilities[:, 0]
            votes[:, class2] += probabilities[:, 1]

        return votes / len(self.pairs)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
