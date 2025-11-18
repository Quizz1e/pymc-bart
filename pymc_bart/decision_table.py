# decision_table.py

#   Copyright 2025 Your Fork
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule
from .pgbart import draw_leaf_value, filter_missing_values  # Reuse from original for leaf drawing

class Level:
    """Level in an oblivious decision tree (all nodes at level use same split)."""
    
    __slots__ = "idx_split_variable", "split_value"
    
    def __init__(self, idx_split_variable: int = -1, split_value: npt.NDArray = np.array([-1.0])):
        self.idx_split_variable = idx_split_variable
        self.split_value = split_value

class DecisionTable:
    """Oblivious decision tree (decision table): balanced, same predicate per level.
    
    Attributes
    ----------
    levels : list[Level]
        List of levels, each with one split variable and value for the entire level.
    leaf_values : npt.NDArray
        Leaf values, shape (2**depth, shape).
    output : npt.NDArray
        Predictions for training data.
    split_rules : list[SplitRule]
        Split rules per variable.
    idx_data_points_per_leaf : list[npt.NDArray[np.int_]] | None
        Data indices per leaf.
    """
    
    __slots__ = ("levels", "leaf_values", "output", "split_rules", "idx_data_points_per_leaf")
    
    def __init__(
        self,
        levels: list[Level],
        leaf_values: npt.NDArray,
        output: npt.NDArray,
        split_rules: list[SplitRule],
        idx_data_points_per_leaf: list[npt.NDArray[np.int_]] | None = None,
    ) -> None:
        self.levels = levels
        self.leaf_values = leaf_values
        self.output = output
        self.split_rules = split_rules
        self.idx_data_points_per_leaf = idx_data_points_per_leaf
    
    @classmethod
    def new_table(
        cls,
        leaf_node_value: npt.NDArray,
        idx_data_points: npt.NDArray[np.int_] | None,
        num_observations: int,
        shape: int,
        split_rules: list[SplitRule],
    ) -> "DecisionTable":
        return cls(
            levels=[],
            leaf_values=leaf_node_value.reshape(1, -1),
            output=np.zeros((num_observations, shape)).astype(config.floatX),
            split_rules=split_rules,
            idx_data_points_per_leaf=[idx_data_points] if idx_data_points is not None else None,
        )
    
    def copy(self) -> "DecisionTable":
        levels_copy = [Level(l.idx_split_variable, l.split_value.copy()) for l in self.levels]
        leaf_values_copy = self.leaf_values.copy()
        idx_dp_copy = [dp.copy() for dp in self.idx_data_points_per_leaf] if self.idx_data_points_per_leaf else None
        return DecisionTable(
            levels=levels_copy,
            leaf_values=leaf_values_copy,
            output=self.output.copy(),
            split_rules=self.split_rules,
            idx_data_points_per_leaf=idx_dp_copy,
        )
    
    def get_depth(self) -> int:
        return len(self.levels)
    
    def grow(
        self,
        idx_split_variable: int,
        split_value: npt.NDArray,
        X: npt.NDArray,
        missing_data: bool,
        sum_trees: npt.NDArray,
        leaf_sd: npt.NDArray,
        m: int,
        response: str,
        normal: object,  # NormalSampler
        shape: int,
    ) -> None:
        """Grow by adding a new level with the same split for all nodes."""
        new_level = Level(idx_split_variable, split_value)
        self.levels.append(new_level)
        
        # Split current leaves
        current_idx_dp = self.idx_data_points_per_leaf if self.idx_data_points_per_leaf is not None else [np.arange(X.shape[0])]
        new_leaf_values = np.zeros((2 * len(current_idx_dp), shape))
        new_idx_dp = [] if self.idx_data_points_per_leaf is not None else None
        
        for leaf_idx, idx_dp in enumerate(current_idx_dp):
            avail_vals = X[idx_dp, idx_split_variable]
            idx_dp, avail_vals = filter_missing_values(avail_vals, idx_dp, missing_data)
            split_rule = self.split_rules[idx_split_variable]
            to_left = split_rule.divide(avail_vals, split_value)
            
            left_idx = idx_dp[to_left]
            right_idx = idx_dp[~to_left]
            
            for side, side_idx in enumerate([left_idx, right_idx]):
                node_value, _ = draw_leaf_value(
                    y_mu_pred=sum_trees[:, side_idx],
                    x_mu=avail_vals[to_left if side == 0 else ~to_left] if response == "linear" else np.array([]),
                    m=m,
                    norm=normal.rvs() * leaf_sd,
                    shape=shape,
                    response=response if response != "mix" else ("linear" if np.random.random() >= 0.5 else "constant"),
                )
                new_leaf_values[2 * leaf_idx + side] = node_value
                if new_idx_dp is not None:
                    new_idx_dp.append(side_idx)
        
        self.leaf_values = new_leaf_values
        self.idx_data_points_per_leaf = new_idx_dp
    
    def prune(self) -> None:
        """Prune by removing the last level."""
        if self.get_depth() == 0:
            return
        self.levels.pop()
        self.leaf_values = self.leaf_values[:self.leaf_values.shape[0] // 2]
        if self.idx_data_points_per_leaf is not None:
            new_idx_dp = []
            for i in range(0, len(self.idx_data_points_per_leaf), 2):
                new_idx_dp.append(np.concatenate((self.idx_data_points_per_leaf[i], self.idx_data_points_per_leaf[i + 1])))
            self.idx_data_points_per_leaf = new_idx_dp
    
    def change_split(self, level_idx: int, new_var: int, new_value: npt.NDArray) -> None:
        """Change split at a specific level."""
        if level_idx >= self.get_depth():
            return
        self.levels[level_idx].idx_split_variable = new_var
        self.levels[level_idx].split_value = new_value
    
    def _predict(self) -> npt.NDArray:
        """Compute predictions for training data."""
        output = self.output
        if self.idx_data_points_per_leaf is not None:
            for leaf_idx, idx_dp in enumerate(self.idx_data_points_per_leaf):
                output[idx_dp] = self.leaf_values[leaf_idx]
        return output.T
    
    def predict(
        self,
        x: npt.NDArray,
        excluded: list[int] | None = None,
        shape: int = 1,
    ) -> npt.NDArray:
        """Predict for new data."""
        if excluded is None:
            excluded = []
        depth = self.get_depth()
        leaf_idx = np.zeros(x.shape[0], dtype=int)
        for d in range(depth):
            if d in excluded:
                continue  # Skip excluded levels (average implicitly by not branching)
            level = self.levels[d]
            to_left = self.split_rules[level.idx_split_variable].divide(x[:, level.idx_split_variable], level.split_value)
            leaf_idx = 2 * leaf_idx + to_left.astype(int)
        return self.leaf_values[leaf_idx]
    
    def get_split_variables(self) -> list[int]:
        """Get list of split variables used."""
        return [level.idx_split_variable for level in self.levels if level.idx_split_variable >= 0]
