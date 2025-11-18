# decision_table.py
# Decision table (catboost-like symmetric trees) implementation
# Designed to be compatible with pymc-bart style tree API,
# but specialized for symmetric trees (the same predicate on every node per level).

# decision_table.py
#   Copyright 2025 Your Team
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

from collections.abc import Generator
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule, ContinuousSplitRule

__all__ = ["DecisionTable", "Node", "get_idx_left_child", "get_idx_right_child", "get_depth"]


class Node:
    """Node of a decision table.

    Attributes
    ----------
    value : npt.NDArray
        For internal nodes: split value (as 1d array); for leaves: leaf value(s).
    nvalue : int
        Number of observations assigned to the node (useful for averaging when excluded).
    idx_data_points : Optional[npt.NDArray[np.int_]]
        Indexes of observations assigned to this node (for leaf nodes).
    idx_split_variable : int
        Feature index used to split at this node (-1 for leaf).
    linear_params : Optional[list[npt.NDArray]]
        For optional linear leaves (kept for API compatibility).
    """

    __slots__ = "value", "nvalue", "idx_data_points", "idx_split_variable", "linear_params"

    def __init__(
        self,
        value: npt.NDArray = np.array([-1.0]),
        nvalue: int = 0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
        linear_params: Optional[List[npt.NDArray]] = None,
    ) -> None:
        self.value = value
        self.nvalue = nvalue
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable
        self.linear_params = linear_params

    @classmethod
    def new_leaf_node(
        cls,
        value: npt.NDArray,
        nvalue: int = 0,
        idx_data_points: Optional[npt.NDArray[np.int_]] = None,
        idx_split_variable: int = -1,
        linear_params: Optional[List[npt.NDArray]] = None,
    ) -> "Node":
        return cls(
            value=value,
            nvalue=nvalue,
            idx_data_points=idx_data_points,
            idx_split_variable=idx_split_variable,
            linear_params=linear_params,
        )

    def is_split_node(self) -> bool:
        return self.idx_split_variable >= 0

    def is_leaf_node(self) -> bool:
        return not self.is_split_node()


def get_idx_left_child(index: int) -> int:
    return index * 2 + 1


def get_idx_right_child(index: int) -> int:
    return index * 2 + 2


@lru_cache
def get_depth(index: int) -> int:
    return (index + 1).bit_length() - 1


class DecisionTable:
    """Symmetric decision table (CatBoost-like).

    This class implements full symmetric trees where every node on the same level
    uses the same split predicate (feature + split value). It aims to be API-compatible
    with :class:`Tree` from the original pymc-bart, but optimized for decision tables
    and for using Metropolis–Hastings moves on level-wise predicates.
    """

    __slots__ = (
        "tree_structure",
        "idx_leaf_nodes",
        "split_rules",
        "depth",
        "output",
    )

    def __init__(
        self,
        tree_structure: dict[int, Node],
        idx_leaf_nodes: List[int],
        split_rules: List[SplitRule],
        depth: int,
        output: npt.NDArray,
    ) -> None:
        self.tree_structure = tree_structure
        self.idx_leaf_nodes = idx_leaf_nodes
        self.split_rules = split_rules
        self.depth = depth
        self.output = output

    @classmethod
    def new_table(
        cls,
        depth: int,
        leaf_node_value: npt.NDArray,
        num_observations: int,
        shape: int,
        split_rules: List[SplitRule],
        feature_per_level: Optional[List[int]] = None,
        split_value_per_level: Optional[List[float]] = None,
    ) -> "DecisionTable":
        """
        Create a symmetric decision table.

        Parameters
        ----------
        depth : int
            Number of split levels (depth >= 1). Leaves = 2**depth.
        leaf_node_value : npt.NDArray
            Initial value for leaves (scalar array or shape-array).
        num_observations : int
            Number of observations; used to build `output` buffer.
        shape : int
            Output shape per observation (1 for scalar regression, >1 for multivariate).
        split_rules : list[SplitRule]
            List of SplitRule objects, one per feature (same API as original).
        feature_per_level : Optional[list[int]]
            If provided, assigns feature index for each level (len == depth).
            If None, idx_split_variable left as -1 (must be set before routing).
        split_value_per_level : Optional[list[float]]
            If provided, assigns split value for each level (len == depth).
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")

        last_index = (2 ** (depth + 1)) - 2
        tree_structure: dict[int, Node] = {}

        for idx in range(last_index + 1):
            tree_structure[idx] = Node.new_leaf_node(value=leaf_node_value, nvalue=0, idx_data_points=None)

        for level in range(depth):
            feat = None
            thr = None
            if feature_per_level is not None:
                feat = int(feature_per_level[level])
            if split_value_per_level is not None:
                thr = float(split_value_per_level[level])
            start = 2**level - 1
            end = 2 ** (level + 1) - 1
            for node_index in range(start, end):
                tree_structure[node_index].idx_split_variable = feat if feat is not None else -1
                tree_structure[node_index].value = np.array([thr]) if thr is not None else np.array([-1.0])

        leaf_start = 2**depth - 1
        idx_leaf_nodes = [i for i in range(leaf_start, leaf_start + 2**depth)]

        output = np.zeros((num_observations, shape)).astype(config.floatX)
        return cls(tree_structure=tree_structure, idx_leaf_nodes=idx_leaf_nodes, split_rules=split_rules, depth=depth, output=output)

    def __getitem__(self, index: int) -> Node:
        return self.get_node(index)

    def __setitem__(self, index: int, node: Node) -> None:
        self.set_node(index, node)

    def copy(self) -> "DecisionTable":
        tree = {
            k: Node(
                value=np.array(v.value),
                nvalue=v.nvalue,
                idx_data_points=None if v.idx_data_points is None else np.array(v.idx_data_points),
                idx_split_variable=v.idx_split_variable,
                linear_params=None if v.linear_params is None else [np.array(p) for p in v.linear_params],
            )
            for k, v in self.tree_structure.items()
        }
        idx_leaf_nodes = self.idx_leaf_nodes.copy() if self.idx_leaf_nodes is not None else None
        out = np.array(self.output) if self.output is not None else None
        return DecisionTable(tree_structure=tree, idx_leaf_nodes=idx_leaf_nodes, split_rules=self.split_rules, depth=self.depth, output=out)

    def trim(self) -> "DecisionTable":
        tree = {
            k: Node(
                value=v.value,
                nvalue=v.nvalue,
                idx_data_points=None,
                idx_split_variable=v.idx_split_variable,
                linear_params=v.linear_params,
            )
            for k, v in self.tree_structure.items()
        }
        return DecisionTable(tree_structure=tree, idx_leaf_nodes=None, split_rules=self.split_rules, depth=self.depth, output=np.array([-1]))

    def get_node(self, index: int) -> Node:
        return self.tree_structure[index]

    def set_node(self, index: int, node: Node) -> None:
        self.tree_structure[index] = node
        if node.is_leaf_node() and self.idx_leaf_nodes is not None:
            if index not in self.idx_leaf_nodes:
                self.idx_leaf_nodes.append(index)

    def get_split_variables(self) -> Generator[int, None, None]:
        for node in self.tree_structure.values():
            if node.is_split_node():
                yield node.idx_split_variable

    def set_level_predicate(self, level: int, feature: int, split_value: float) -> None:
        """Set (feature, split_value) for all internal nodes at `level` (0-based)."""
        start = 2**level - 1
        end = 2 ** (level + 1) - 1
        for node_index in range(start, end):
            node = self.tree_structure[node_index]
            node.idx_split_variable = int(feature)
            node.value = np.array([float(split_value)])

    def get_level_predicate(self, level: int) -> Tuple[Optional[int], Optional[float]]:
        """Return (feature, split_value) if identical across level, else (None, None)."""
        start = 2**level - 1
        end = 2 ** (level + 1) - 1
        feats = []
        thrs = []
        for node_index in range(start, end):
            node = self.tree_structure[node_index]
            feats.append(node.idx_split_variable)
            thrs.append(float(node.value[0]) if node.value is not None else None)
        feat_unique = feats[0] if all(f == feats[0] for f in feats) else None
        thr_unique = thrs[0] if all(t == thrs[0] for t in thrs) else None
        return feat_unique, thr_unique

    def route_data(self, X: npt.NDArray) -> None:
        """
        Route rows of X to leaves and fill node.idx_data_points and node.nvalue.
        Requires that every level has concrete (feature, threshold) set.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_obs, n_features)")

        n, p = X.shape

        features = []
        thresholds = []
        for level in range(self.depth):
            feat, thr = self.get_level_predicate(level)
            if feat is None or thr is None:
                raise ValueError("Level predicates must be set prior to routing (use set_level_predicate).")
            features.append(int(feat))
            thresholds.append(float(thr))

        bits = np.zeros((n,), dtype=np.int64)
        for feat, thr in zip(features, thresholds):
            left_mask = self.split_rules[feat].divide(X[:, feat], np.array([thr]))
            to_right = (~left_mask).astype(np.int64)
            bits = bits * 2 + to_right

        leaf_offset = 2**self.depth - 1

        for li in self.idx_leaf_nodes:
            self.tree_structure[li].idx_data_points = np.array([], dtype=np.int32)
            self.tree_structure[li].nvalue = 0

        if bits.size > 0:
            for b in range(2**self.depth):
                mask = bits == b
                idxs = np.nonzero(mask)[0].astype(np.int32)
                node_index = leaf_offset + b
                self.tree_structure[node_index].idx_data_points = idxs
                self.tree_structure[node_index].nvalue = idxs.size

        if self.output is not None:
            for node_index in self.idx_leaf_nodes:
                node = self.tree_structure[node_index]
                if node.idx_data_points is not None and node.value is not None:
                    vals = np.asarray(node.value)
                    vals = vals.reshape(-1)
                    if vals.size == 1:
                        self.output[node.idx_data_points, :] = vals[0]
                    else:
                        self.output[node.idx_data_points, :] = vals

    def _predict(self) -> npt.NDArray:
        output = self.output
        if self.idx_leaf_nodes is not None:
            for node_index in self.idx_leaf_nodes:
                leaf_node = self.get_node(node_index)
                if leaf_node.idx_data_points is None:
                    continue
                output[leaf_node.idx_data_points] = leaf_node.value
        return output.T

    def predict(self, x: npt.NDArray, excluded: Optional[List[int]] = None, shape: int = 1) -> npt.NDArray:
        """
        Predict values for input x.

        If excluded is None (fast path) we use vectorized routing and return array of shape (shape, n_obs).
        If excluded is provided we fall back to traversal that supports excluded semantics similar to Tree._traverse_tree.
        """
        if excluded is None or len(excluded) == 0:
            single = False
            if x.ndim == 1:
                x = x.reshape(1, -1)
                single = True
            n, p = x.shape

            features = []
            thresholds = []
            for level in range(self.depth):
                feat, thr = self.get_level_predicate(level)
                if feat is None or thr is None:
                    return self._traverse_tree(X=x, excluded=excluded, shape=shape)
                features.append(int(feat))
                thresholds.append(float(thr))

            bits = np.zeros((n,), dtype=np.int64)
            for feat, thr in zip(features, thresholds):
                left_mask = self.split_rules[feat].divide(x[:, feat], np.array([thr]))
                to_right = (~left_mask).astype(np.int64)
                bits = bits * 2 + to_right

            leaf_offset = 2**self.depth - 1
            out = np.zeros((shape, n)).astype(config.floatX)
            for i in range(n):
                node_index = leaf_offset + int(bits[i])
                val = np.asarray(self.tree_structure[node_index].value).reshape(-1)
                if val.size == 1:
                    out[:, i] = val[0]
                else:
                    if val.size != shape:
                        raise ValueError("Leaf value size does not match requested shape")
                    out[:, i] = val
            if single:
                return out[:, 0]
            return out
        else:
            return self._traverse_tree(X=x, excluded=excluded, shape=shape)

    def _traverse_tree(
        self,
        X: npt.NDArray,
        excluded: Optional[List[int]] = None,
        shape: int | tuple[int, ...] = 1,
    ) -> npt.NDArray:
        """
        Generic traversal supporting `excluded` semantics.
        Implementation mirrors Tree._traverse_tree from pymc-bart but adapted for symmetric table.
        """
        x_shape = (1,) if len(X.shape) == 1 else X.shape[:-1]
        nd_dims = (...,) + (None,) * len(x_shape)

        stack: List[tuple[int, npt.NDArray, int]] = [(0, np.ones(x_shape), 0)]
        p_d = np.zeros(shape + x_shape) if isinstance(shape, tuple) else np.zeros((shape,) + x_shape)
        while stack:
            node_index, weights, idx_split_variable = stack.pop()
            node = self.get_node(node_index)
            if node.is_leaf_node():
                params = node.linear_params
                if params is None:
                    p_d += weights * node.value[nd_dims]
                else:
                    p_d += weights * (params[0][nd_dims] + params[1][nd_dims] * X[..., idx_split_variable])
            else:
                idx_split_variable = node.idx_split_variable
                left_node_index, right_node_index = get_idx_left_child(node_index), get_idx_right_child(node_index)
                if excluded is not None and idx_split_variable in excluded:
                    prop_nvalue_left = self.get_node(left_node_index).nvalue / node.nvalue if node.nvalue > 0 else 0.5
                    stack.append((left_node_index, weights * prop_nvalue_left, idx_split_variable))
                    stack.append((right_node_index, weights * (1 - prop_nvalue_left), idx_split_variable))
                else:
                    to_left = self.split_rules[idx_split_variable].divide(X[..., idx_split_variable], node.value).astype("float")
                    stack.append((left_node_index, weights * to_left, idx_split_variable))
                    stack.append((right_node_index, weights * (1 - to_left), idx_split_variable))

        if len(X.shape) == 1:
            p_d = p_d[..., 0]

        return p_d

    def get_leaf_values(self) -> List[npt.NDArray]:
        """Return list of leaf values ordered by idx_leaf_nodes."""
        return [self.tree_structure[i].value for i in self.idx_leaf_nodes]

    def set_leaf_values(self, values: List[npt.NDArray]) -> None:
        """Set leaf values; input list must match number of leaves."""
        if len(values) != len(self.idx_leaf_nodes):
            raise ValueError("Length of values must match number of leaves")
        for node_index, v in zip(self.idx_leaf_nodes, values):
            self.tree_structure[node_index].value = np.array(v)

    def draw_leaf_values_posterior(
        self,
        residual: npt.NDArray,
        sigma: float,
        tau: float,
        prior_mean: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[npt.NDArray]:
        """
        Draw posterior leaf values conditionally (Gaussian likelihood + Gaussian prior).

        residual : Y - sum_other_trees (1D array length n_obs)
        sigma : noise std
        tau : prior std for leaf value
        prior_mean : prior mean
        Returns list of drawn leaf values in order of idx_leaf_nodes.
        """
        if rng is None:
            rng = np.random.default_rng()

        drawn: List[npt.NDArray] = []
        for idx in self.idx_leaf_nodes:
            node = self.tree_structure[idx]
            obs_idx = node.idx_data_points
            if obs_idx is None or obs_idx.size == 0:
                var_post = tau**2
                mean_post = prior_mean
            else:
                y_leaf = residual[obs_idx]
                n = y_leaf.size
                denom = (n / (sigma**2)) + (1.0 / (tau**2))
                var_post = 1.0 / denom
                mean_part = (np.sum(y_leaf) / (sigma**2)) + (prior_mean / (tau**2))
                mean_post = var_post * mean_part
            draw = rng.normal(loc=mean_post, scale=np.sqrt(var_post))
            node.value = np.array([draw]) if np.isscalar(draw) else np.array(draw)
            drawn.append(node.value)
        return drawn
