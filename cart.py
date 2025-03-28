import numpy as np

class CARTClassifier:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(sample, self.tree) for sample in X])

    def _gini(self, y):
        # 计算基尼指数
        m = len(y)
        if m == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / m
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X, y, feature, threshold):
        # 根据特定特征和阈值分裂数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_best_split(self, X, y):
        # 寻找最佳分裂点
        m, n_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        best_splits = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
                gini_left, gini_right = self._gini(y_left), self._gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature, best_threshold = feature, threshold
                    best_splits = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_splits

    def _build_tree(self, X, y, depth):
        # 递归构建决策树
        num_samples, num_classes = len(y), len(np.unique(y))
        if depth >= self.max_depth or num_samples < self.min_samples_split or num_classes == 1:
            leaf_value = np.bincount(y).argmax()
            return {"type": "leaf", "class": leaf_value}

        feature, threshold, splits = self._find_best_split(X, y)
        if not splits:
            leaf_value = np.bincount(y).argmax()
            return {"type": "leaf", "class": leaf_value}

        X_left, X_right, y_left, y_right = splits
        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1)
        }

    def _predict_one(self, sample, node):
        # 单个样本的预测
        if node["type"] == "leaf":
            return node["class"]
        if sample[node["feature"]] <= node["threshold"]:
            return self._predict_one(sample, node["left"])
        else:
            return self._predict_one(sample, node["right"])

# 测试代码
if __name__ == "__main__":
    # 示例数据集
    X = np.array([
        [2.771244718, 1.784783929],
        [1.728571309, 1.169761413],
        [3.678319846, 2.81281357],
        [3.961043357, 2.61995032],
        [2.999208922, 2.209014212],
        [7.497545867, 3.162953546],
        [9.00220326, 3.339047188],
        [7.444542326, 0.476683375],
        [10.12493903, 3.234550982],
        [6.642287351, 3.319983761],
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # 创建并训练模型
    clf = CARTClassifier(max_depth=3, min_samples_split=2)
    clf.fit(X, y)

    # 测试模型
    predictions = clf.predict(X)
    print(clf.tree)
    print("预测结果:", predictions)
    print("真实标签:", y)
