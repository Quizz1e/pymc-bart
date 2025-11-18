# decision_table.py
# Copyright 2022 The PyMC Developers + ваши изменения
# Apache License 2.0

import numpy as np
import numpy.typing as npt
from pytensor import config
from .split_rules import SplitRule

class DecisionTable:
    __slots__ = ["levels", "leaf_values", "leaf_nvalues", "idx_data_points_per_leaf",
                 "split_rules", "output", "leaves_shape"]

    def __init__(self, levels, leaf_values, leaf_nvalues, idx_data_points_per_leaf,
                 split_rules, output, leaves_shape):
        self.levels = levels  # list[(feature_idx, split_value)]
        self.leaf_values = leaf_values
        self.leaf_nvalues = leaf_nvalues
        self.idx_data_points_per_leaf = idx_data_points_per_leaf
        self.split_rules = split_rules
        self.output = output
        self.leaves_shape = leaves_shape

    @classmethod
    def new_tree(cls, leaf_node_value: npt.NDArray, idx_data_points: npt.NDArray,
                 num_observations: int, leaves_shape: int, split_rules: list[SplitRule]):
        leaf_values = np.full((1, leaves_shape), leaf_node_value, dtype=config.floatX)
        leaf_nvalues = np.array([len(idx_data_points)])
        idx_per_leaf = [idx_data_points]
        output = np.zeros((num_observations, leaves_shape), dtype=config.floatX)
        return cls([], leaf_values, leaf_nvalues, idx_per_leaf, split_rules, output, leaves_shape)

    def copy(self):
        return DecisionTable(
            self.levels.copy(),
            self.leaf_values.copy(),
            self.leaf_nvalues.copy(),
            [a.copy() for a in self.idx_data_points_per_leaf],
            self.split_rules,
            self.output.copy(),
            self.leaves_shape,
        )

    def trim(self):
        return DecisionTable(
            self.levels.copy(),
            self.leaf_values.copy(),
            self.leaf_nvalues.copy(),
            None, self.split_rules, np.array([-1.0]), self.leaves_shape
        )

    def get_split_variables(self):
        for feature, _ in self.levels:
            yield feature

    def _predict(self):
        self.output.fill(0.0)
        for leaf, idxs in enumerate(self.idx_data_points_per_leaf):
            if len(idxs) > 0:
                self.output[idxs] = self.leaf_values[leaf]
        return self.output.T

    def predict(self, X: npt.NDArray, excluded=None, shape=None):
        if excluded:
            raise NotImplementedError("excluded not supported yet")
        if shape is None:
            shape = self.leaves_shape
        n = X.shape[0]
        leaf_idx = np.zeros(n, dtype=int)
        for var_idx, split_value in self.levels:
            to_left = self.split_rules[var_idx].divide(X[:, var_idx], split_value)
            leaf_idx = leaf_idx * 2 + (1 - to_left.astype(int))
        return self.leaf_values[leaf_idx]

    def update_idx_data_points(self, X: npt.NDArray, missing_data: bool):
        n = X.shape[0]
        leaf_idx = np.zeros(n, dtype=int)
        for var_idx, split_value in self.levels:
            vals = X[:, var_idx]
            to_left = np.zeros(n, dtype=bool)
            if missing_data:
                mask = ~np.isnan(vals)
                to_left[mask] = self.split_rules[var_idx].divide(vals[mask], split_value)
            else:
                to_left = self.split_rules[var_idx].divide(vals, split_value)
            leaf_idx = leaf_idx * 2 + (1 - to_left.astype(int))
        num_leaves = 1 << len(self.levels)
        self.idx_data_points_per_leaf = [np.where(leaf_idx == i)[0] for i in range(num_leaves)]
        self.leaf_nvalues = np.array([len(idxs) for idxs in self.idx_data_points_per_leaf])
