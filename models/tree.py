import numpy as np
from models.abstract_model import Model


class DecisionTreeClassifier(Model):
    def __init__(self, max_depth=None, min_samples_split=2, method="cart"):
        self.max_depth = max_depth
        self.method = method
        self.min_samples_split = min_samples_split

    def fit(self, X, y, dtype_dict):
        self.classes_ = np.unique(y)
        self.dtype_dict = list(dtype_dict.values())
        self.y = y
        if self.method == "c4.5":
            self.tree = self._build_tree_c45(X, y)
        elif self.method == "cart":
            self.tree = self._build_tree_cart(X, y)
        elif self.method == "id3":
            self.tree = self._build_tree_id3(X, y)
        else:
            raise NotImplementedError(
                "This method is not implement! try other methods (c4.5,cart,id3)"
            )

    def _build_tree_id3(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or n_classes == 1
        ):
            classes, class_counts = np.unique(y, return_counts=True)
            return {"label": classes[np.argmax(class_counts)]}

        best_info_gain = -np.inf
        best_feature_idx = None
        best_threshold = None

        target_class_entropy = self.entropy(y)
        for feature_idx in range(n_features):
            feature_type = self.dtype_dict[feature_idx]

            if feature_type == "category":
                categories, counts = np.unique(X[:, feature_idx], return_counts=True)
                entropy_info = 0
                for category, count in zip(categories, counts):
                    y_subset = y[X[:, feature_idx] == category]
                    entropy_info += (count / n_samples) * self.entropy(y_subset)

                information_gain = target_class_entropy - entropy_info

                if information_gain > best_info_gain:
                    best_info_gain = information_gain
                    best_feature_idx = feature_idx
                    best_threshold = None

            else:
                sorted_indices = np.argsort(X[:, feature_idx])
                sorted_y = y[sorted_indices]
                sorted_X = X[sorted_indices, feature_idx]

                for i in range(1, n_samples):
                    if sorted_X[i] == sorted_X[i - 1]:
                        continue
                    threshold = (sorted_X[i] + sorted_X[i - 1]) / 2
                    y_left = sorted_y[:i]
                    y_right = sorted_y[i:]
                    entropy_left = self.entropy(y_left)
                    entropy_right = self.entropy(y_right)
                    p_left = i / n_samples
                    p_right = 1 - p_left

                    entropy_info = p_left * entropy_left + p_right * entropy_right
                    information_gain = target_class_entropy - entropy_info

                    if information_gain > best_info_gain:
                        best_info_gain = information_gain
                        best_feature_idx = feature_idx
                        best_threshold = threshold

        if best_threshold is not None:
            left_indices = X[:, best_feature_idx] <= best_threshold
            right_indices = X[:, best_feature_idx] > best_threshold
            left_subtree = self._build_tree_id3(
                X[left_indices], y[left_indices], depth + 1
            )
            right_subtree = self._build_tree_id3(
                X[right_indices], y[right_indices], depth + 1
            )
            return {
                "feature_idx": best_feature_idx,
                "threshold": best_threshold,
                "subtrees": {"<=": left_subtree, ">": right_subtree},
            }
        else:
            sub_trees = {}
            for category in np.unique(X[:, best_feature_idx]):
                category_indices = X[:, best_feature_idx] == category
                sub_trees[category] = self._build_tree_id3(
                    X[category_indices], y[category_indices], depth + 1
                )
            return {"feature_idx": best_feature_idx, "subtrees": sub_trees}

    def _build_tree_c45(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or n_classes == 1
        ):
            classes, class_counts = np.unique(y, return_counts=True)
            return {"label": classes[np.argmax(class_counts)]}

        best_gain_ratio = -np.inf
        best_feature_idx = None
        best_threshold = None

        target_class_entropy = self.entropy(y)
        for feature_idx in range(n_features):
            feature_type = self.dtype_dict[feature_idx]

            if feature_type == "category":
                categories, counts = np.unique(X[:, feature_idx], return_counts=True)
                entropy_info = 0
                for category, count in zip(categories, counts):
                    y_subset = y[X[:, feature_idx] == category]
                    entropy_info += (count / n_samples) * self.entropy(y_subset)

                split_info = -np.sum(
                    (counts / n_samples) * np.log2(counts / n_samples + (counts == 0))
                )
                information_gain = target_class_entropy - entropy_info
                gain_ratio = information_gain / split_info if split_info > 0 else 0

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature_idx = feature_idx
                    best_threshold = None

            else:
                sorted_indices = np.argsort(X[:, feature_idx])
                sorted_y = y[sorted_indices]
                sorted_X = X[sorted_indices, feature_idx]

                for i in range(1, n_samples):
                    if sorted_X[i] == sorted_X[i - 1]:
                        continue
                    threshold = (sorted_X[i] + sorted_X[i - 1]) / 2
                    y_left = sorted_y[:i]
                    y_right = sorted_y[i:]
                    entropy_left = self.entropy(y_left)
                    entropy_right = self.entropy(y_right)
                    p_left = i / n_samples
                    p_right = 1 - p_left

                    entropy_info = p_left * entropy_left + p_right * entropy_right
                    split_info = -(
                        p_left * np.log2(p_left) + p_right * np.log2(p_right)
                    )

                    information_gain = target_class_entropy - entropy_info
                    gain_ratio = information_gain / split_info if split_info > 0 else 0

                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feature_idx = feature_idx
                        best_threshold = threshold

        if best_threshold is not None:
            left_indices = X[:, best_feature_idx] <= best_threshold
            right_indices = X[:, best_feature_idx] > best_threshold
            left_subtree = self._build_tree_id3(
                X[left_indices], y[left_indices], depth + 1
            )
            right_subtree = self._build_tree_id3(
                X[right_indices], y[right_indices], depth + 1
            )
            return {
                "feature_idx": best_feature_idx,
                "threshold": best_threshold,
                "subtrees": {"<=": left_subtree, ">": right_subtree},
            }
        else:
            sub_trees = {}
            for category in np.unique(X[:, best_feature_idx]):
                category_indices = X[:, best_feature_idx] == category
                sub_trees[category] = self._build_tree_id3(
                    X[category_indices], y[category_indices], depth + 1
                )
            return {"feature_idx": best_feature_idx, "subtrees": sub_trees}

    def _build_tree_cart(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or n_classes == 1
        ):
            classes, class_counts = np.unique(y, return_counts=True)
            probs = np.zeros(len(self.classes_))
            for class_, count in zip(classes, class_counts):
                probs[class_] = count / n_samples
            return {"label": classes[np.argmax(class_counts)], "probs": probs}

        best_gini_gain = -np.inf
        best_feature_idx = None
        best_threshold = None

        parent_gini_impurity = self.gini(y)
        for feature_idx in range(n_features):
            feature_type = self.dtype_dict[feature_idx]
            if feature_type == "category":
                categories, counts = np.unique(X[:, feature_idx], return_counts=True)
                gini_impurity = 0
                for category, count in zip(categories, counts):
                    y_subset = y[X[:, feature_idx] == category]
                    gini_impurity += (count / n_samples) * self.gini(y_subset)

                gini_gain = parent_gini_impurity - gini_impurity
                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_feature_idx = feature_idx
                    best_threshold = None

            else:
                sorted_indices = np.argsort(X[:, feature_idx])
                sorted_y = y[sorted_indices]
                sorted_X = X[sorted_indices, feature_idx]

                for i in range(1, n_samples):
                    if sorted_X[i] == sorted_X[i - 1]:
                        continue
                    threshold = (sorted_X[i] + sorted_X[i - 1]) / 2
                    y_left = sorted_y[:i]
                    y_right = sorted_y[i:]
                    p_left = i / n_samples
                    p_right = 1 - p_left

                    gini_left = self.gini(y_left)
                    gini_right = self.gini(y_right)

                    weighted_gini_impurity = p_left * gini_left + p_right * gini_right
                    gini_gain = parent_gini_impurity - weighted_gini_impurity

                    if gini_gain > best_gini_gain:
                        best_gini_gain = gini_gain
                        best_feature_idx = feature_idx
                        best_threshold = threshold

        if best_feature_idx is None:
            classes, class_counts = np.unique(y, return_counts=True)
            probs = np.zeros(len(self.classes_))
            for class_, count in zip(classes, class_counts):
                probs[class_] = count / n_samples
            return {"label": classes[np.argmax(class_counts)], "probs": probs}

        if best_threshold is not None:
            left_indices = X[:, best_feature_idx] <= best_threshold
            right_indices = X[:, best_feature_idx] > best_threshold
            left_subtree = self._build_tree_cart(
                X[left_indices], y[left_indices], depth + 1
            )
            right_subtree = self._build_tree_cart(
                X[right_indices], y[right_indices], depth + 1
            )
            return {
                "feature_idx": best_feature_idx,
                "threshold": best_threshold,
                "subtrees": {"<=": left_subtree, ">": right_subtree},
            }
        else:
            sub_trees = {}
            for category in np.unique(X[:, best_feature_idx]):
                category_indices = X[:, best_feature_idx] == category
                sub_trees[category] = self._build_tree_cart(
                    X[category_indices], y[category_indices], depth + 1
                )
            return {"feature_idx": best_feature_idx, "subtrees": sub_trees}

    @staticmethod
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if "label" in tree:
            return tree["label"]
        if "threshold" in tree:
            subtree_key = "<=" if x[tree["feature_idx"]] <= tree["threshold"] else ">"
            subtree = tree["subtrees"].get(subtree_key)
        else:
            subtree = tree["subtrees"].get(x[tree["feature_idx"]], None)

        if subtree is None:
            classes, class_counts = np.unique(self.y, return_counts=True)
            return classes[np.argmax(class_counts)]

        return self._predict_tree(x, subtree)

    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.tree
        if "label" in tree:
            print(indent + "Leaf: Predict = {}".format(tree["label"]))
        else:

            if "threshold" in tree:
                print(
                    indent
                    + "Node: Feature {} (Threshold = {:.3f})".format(
                        tree["feature_idx"], tree["threshold"]
                    )
                )
                for value, subtree in tree["subtrees"].items():
                    print(indent + "  Value {}:".format(value))
                    self.print_tree(subtree, indent + "    ")
            else:

                print(indent + "Node: Feature {}".format(tree["feature_idx"]))
                for value, subtree in tree["subtrees"].items():
                    print(indent + "  Value {}:".format(value))
                    self.print_tree(subtree, indent + "    ")

    def predict_proba(self, X):
        return np.array([self._predict_tree_proba(x, self.tree) for x in X])

    def _predict_tree_proba(self, x, tree):
        if "probs" in tree:
            return tree["probs"]
        if "threshold" in tree:
            subtree_key = "<=" if x[tree["feature_idx"]] <= tree["threshold"] else ">"
            subtree = tree["subtrees"].get(subtree_key)
        else:
            subtree = tree["subtrees"].get(x[tree["feature_idx"]], None)

        if subtree is None:
            return np.zeros(len(self.classes_))

        return self._predict_tree_proba(x, subtree)
