# decision_table.py
# Decision table (catboost-like symmetric trees) implementation
# Designed to be compatible with pymc-bart style tree API,
# but specialized for symmetric trees (the same predicate on every node per level)

# pymc_bart/decision_table.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule


class DecisionTable:
    """
    Oblivious decision table = полностью симметричное дерево,
    где на каждом уровне используется одна и та же переменная и один и тот же порог.
    """
    __slots__ = ("depth", "features", "thresholds", "leaf_values", "split_rules", "shape")

    def __init__(self, split_rules: list[SplitRule], shape: int = 1):
        self.split_rules = split_rules
        self.shape = shape

        self.depth = 0
        self.features: list[int] = []
        self.thresholds: list[float] = []
        self.leaf_values: npt.NDArray = np.zeros((1, shape), dtype=config.floatX)

    def copy(self) -> "DecisionTable":
        new = DecisionTable(self.split_rules, self.shape)
        new.depth = self.depth
        new.features = self.features.copy()
        new.thresholds = self.thresholds.copy()
        new.leaf_values = self.leaf_values.copy()
        return new

    def num_leaves(self) -> int:
        return 1 << self.depth                      # 2**depth

    def get_leaf_indices(self, X: npt.NDArray) -> npt.NDArray:
        """Возвращает индекс листа для каждой наблюдения."""
        n = X.shape[0]
        indices = np.zeros(n, dtype=np.int32)

        for level in range(self.depth):
            feat = self.features[level]
            thr = self.thresholds[level]
            to_right = self.split_rules[feat].divide(X[:, feat], thr)
            indices = indices * 2 + to_right.astype(np.int32)

        return indices

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Предсказание дерева на матрице X."""
        if self.depth == 0:
            return np.tile(self.leaf_values, (X.shape[0], 1))

        leaf_idx = self.get_leaf_indices(X)
        return self.leaf_values[leaf_idx]

    def grow(self, feature: int, threshold: float):
        """Добавить один уровень."""
        self.features.append(feature)
        self.thresholds.append(threshold)
        self.depth += 1
        # дублируем текущие значения листьев
        self.leaf_values = np.repeat(self.leaf_values, 2, axis=0)

    def prune(self):
        """Убрать последний уровень (оставляем только чётные листья)."""
        if self.depth == 0:
            return
        self.features.pop()
        self.thresholds.pop()
        self.depth -= 1
        self.leaf_values = self.leaf_values[::2].copy()

    def change_level(self, level: int, new_feature: int, new_threshold: float):
        """Сменить фичу и порог на конкретном уровне."""
        self.features[level] = new_feature
        self.thresholds[level] = new_threshold
