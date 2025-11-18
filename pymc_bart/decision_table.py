# decision_table.py
# Decision table (catboost-like symmetric trees) implementation
# Designed to be compatible with pymc-bart style tree API,
# but specialized for symmetric trees (the same predicate on every node per level)

# decision_table.py
from typing import List

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule

class DecisionTable:
    """Oblivious decision table (одна фича и один порог на уровне, полностью сбалансированное дерево)."""
    __slots__ = ("depth", "features", "thresholds", "leaf_values", "split_rules", "shape")

    def __init__(self, split_rules: List[SplitRule], shape: int = 1):
        self.split_rules = split_rules
        self.shape = shape
        self.depth = 0
        self.features: List[int] = []
        self.thresholds: List[float] = []
        self.leaf_values: npt.NDArray = np.zeros((1, shape), dtype=config.floatX)

    def copy(self) -> "DecisionTable":
        new = DecisionTable(self.split_rules, self.shape)
        new.depth = self.depth
        new.features = self.features.copy()
        new.thresholds = self.thresholds.copy()
        new.leaf_values = self.leaf_values.copy()
        return new

    def get_num_leaves(self) -> int:
        return 1 << self.depth  # 2 ** depth

    def get_leaf_indices(self, X: npt.NDArray) -> npt.NDArray:
        n = X.shape[0]
        indices = np.zeros(n, dtype=np.int32)
        for level in range(self.depth):
            feat = self.features[level]
            thr = self.thresholds[level]
            # True → right child
            right = self.split_rules[feat].divide(X[:, feat], thr).ravel()
            indices = indices * 2
            indices[right] += 1
        return indices

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if self.depth == 0:
            return np.tile(self.leaf_values, (X.shape[0], 1))

        indices = self.get_leaf_indices(X)
        return self.leaf_values[indices]

    def grow(self, feature: int, threshold: float):
        """Добавить уровень (вызывается только внутри сэмплера)."""
        self.features.append(feature)
        self.thresholds.append(threshold)
        self.depth += 1
        # повторяем старые значения листьев — потом перезапишем новыми
        self.leaf_values = np.repeat(self.leaf_values, 2, axis=0)

    def prune(self):
        """Убрать последний уровень."""
        if self.depth == 0:
            return
        self.features.pop()
        self.thresholds.pop()
        self.depth -= 1
        # временно оставляем чётные листья — потом перезапишем
        self.leaf_values = self.leaf_values[::2]
