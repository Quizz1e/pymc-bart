# src/pymc_bart/decision_table.py
import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule


class DecisionTable:
    """Oblivious decision table (CatBoost-style tree) для BART."""
    __slots__slots__ = ("splits", "leaf_values", "leaf_idx", "split_rules", "depth", "m", "leaves_shape")

    def __init__(
        self,
        splits: list[tuple[int, float]],
        leaf_values: npt.NDArray,
        leaf_idx: list[npt.NDArray],
        split_rules: list[SplitRule],
        m: int,
        leaves_shape: int,
    ):
        self.splits = splits
        self.leaf_values = leaf_values.astype(config.floatX)
        self.leaf_idx = leaf_idx
        self.split_rules = split_rules
        self.depth = len(splits)
        self.m = m
        self.leaves_shape = leaves_shape
        assert len(leaf_values) == 2 ** self.depth
        assert len(leaf_idx) == 2 ** self.depth

    @classmethod
    def initial(cls, init_val: float, n_obs: int, leaves_shape: int, split_rules: list[SplitRule], m: int):
        leaf_values = np.full((1, leaves_shape), init_val, dtype=config.floatX)
        leaf_idx = [np.arange(n_obs, dtype=np.int32)]
        return cls([], leaf_values, leaf_idx, split_rules, m, leaves_shape)

    def copy(self):
        return DecisionTable(
            splits=self.splits.copy(),
            leaf_values=self.leaf_values.copy(),
            leaf_idx=[arr.copy() for arr in self.leaf_idx],
            split_rules=self.split_rules,
            m=self.m,
            leaves_shape=self.leaves_shape,
        )

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """O(n × depth) предсказание."""
        if self.depth == 0:
            return np.full((X.shape[0], self.leaves_shape), self.leaf_values[0])

        leaf_idx = np.zeros(X.shape[0], dtype=np.int32)
        power = 1
        for var_idx, split_val in self.splits:
            to_left = self.split_rules[var_idx].divide(X[:, var_idx], split_val)
            leaf_idx += to_left.astype(np.int32) * power
            power <<= 1
        return self.leaf_values[leaf_idx]

    def get_split_variables(self) -> set[int]:
        return {var_idx for var_idx, _ in self.splits}

    def _grow(self, var_idx: int, split_val: float) -> "DecisionTable | None":
        new_n = len(self.leaf_idx) * 2
        new_values = np.empty((new_n, self.leaves_shape), dtype=config.floatX)
        new_idx = []

        for i, idxs in enumerate(self.leaf_idx):
            values = self.X[idxs, var_idx]
            mask = self.split_rules[var_idx].divide(values, split_val)
            left = idxs[mask]
            right = idxs[~mask]
            if len(left) == 0 or len(right) == 0:
                return None
            new_idx.extend([left, right])
            new_values[2*i] = self.leaf_values[i]
            new_values[2*i+1] = self.leaf_values[i]

        return DecisionTable(
            splits=self.splits + [(var_idx, split_val)],
            leaf_values=new_values,
            leaf_idx=new_idx,
            split_rules=self.split_rules,
            m=self.m,
            leaves_shape=self.leaves_shape,
        )

    def _prune(self) -> "DecisionTable":
        if self.depth == 0:
            return self
        new_n = len(self.leaf_idx) // 2
        new_values = np.empty((new_n, self.leaves_shape), dtype=config.floatX)
        new_idx = []
        for i in range(new_n):
            new_idx.append(np.concatenate([self.leaf_idx[2*i], self.leaf_idx[2*i+1]))
            new_values[i] = self.leaf_values[2*i]  # не важно, будет перезаписано
        return DecisionTable(
            splits=self.splits[:-1],
            leaf_values=new_values,
            leaf_idx=new_idx,
            split_rules=self.split_rules,
            m=self.m,
            leaves_shape=self.leaves_shape,
        )
