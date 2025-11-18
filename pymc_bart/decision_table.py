import numpy as np
import numpy.typing as npt
from numba import njit
from pytensor import config

from .split_rules import SplitRule


class DecisionTable:
    """
    Symmetric tree (decision table) where all nodes at the same level 
    use the same predicate (split variable and value).
    """
    
    __slots__ = ("depth", "split_variables", "split_values", "leaf_values", "output", "split_rules")
    
    def __init__(
        self,
        depth: int,
        split_variables: npt.NDArray[np.int_],
        split_values: npt.NDArray,
        leaf_values: npt.NDArray,
        output: npt.NDArray,
        split_rules: list[SplitRule],
    ):
        self.depth = depth
        self.split_variables = split_variables  # shape: (depth,)
        self.split_values = split_values        # shape: (depth,)
        self.leaf_values = leaf_values          # shape: (2^depth, shape)
        self.output = output
        self.split_rules = split_rules
    
    @classmethod
    def new_table(
        cls,
        leaf_node_value: npt.NDArray,
        num_observations: int,
        shape: int,
        split_rules: list[SplitRule],
        max_depth: int = 5,
    ) -> "DecisionTable":
        depth = 0
        split_variables = np.full(max_depth, -1, dtype=np.int32)
        split_values = np.full(max_depth, -1.0)
        leaf_values = np.full((2 ** max_depth, shape), leaf_node_value)
        
        return cls(
            depth=depth,
            split_variables=split_variables,
            split_values=split_values,
            leaf_values=leaf_values,
            output=np.zeros((num_observations, shape)).astype(config.floatX),
            split_rules=split_rules,
        )
    
    def copy(self) -> "DecisionTable":
        return DecisionTable(
            depth=self.depth,
            split_variables=self.split_variables.copy(),
            split_values=self.split_values.copy(),
            leaf_values=self.leaf_values.copy(),
            output=self.output,
            split_rules=self.split_rules,
        )
    
    def trim(self) -> "DecisionTable":
        return DecisionTable(
            depth=self.depth,
            split_variables=self.split_variables.copy(),
            split_values=self.split_values.copy(),
            leaf_values=self.leaf_values.copy(),
            output=np.array([-1]),
            split_rules=self.split_rules,
        )
    
    def _predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict using the decision table."""
        n_samples = X.shape[0]
        n_shape = self.leaf_values.shape[1]
        
        predictions = np.zeros((n_samples, n_shape))
        
        for i in range(n_samples):
            leaf_idx = self._traverse(X[i])
            predictions[i] = self.leaf_values[leaf_idx]
        
        return predictions.T
    
    def _traverse(self, x: npt.NDArray) -> int:
        """Traverse the decision table for a single sample."""
        leaf_idx = 0
        for level in range(self.depth):
            split_var = self.split_variables[level]
            split_val = self.split_values[level]
            
            if split_var == -1:  # Not split at this level yet
                continue
                
            rule = self.split_rules[split_var]
            goes_left = rule.divide(np.array([x[split_var]]), split_val)[0]
            
            # Move to left (0) or right (1) child
            leaf_idx = leaf_idx * 2 + (0 if goes_left else 1)
        
        return leaf_idx
    
    def predict(
        self,
        x: npt.NDArray,
        excluded: list[int] | None = None,
        shape: int = 1,
    ) -> npt.NDArray:
        """Predict output for input x."""
        if excluded is None:
            excluded = []
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        return self._predict(x)
    
    def get_split_variables(self) -> list[int]:
        """Get all split variables used in the table."""
        return [var for var in self.split_variables[:self.depth] if var != -1]
    
    def grow_level(
        self,
        level: int,
        split_variable: int,
        split_value: float,
        new_leaf_values: npt.NDArray,
    ) -> None:
        """Grow the table by adding a split at the specified level."""
        if level >= len(self.split_variables):
            raise ValueError(f"Cannot grow beyond max depth {len(self.split_variables)}")
        
        self.split_variables[level] = split_variable
        self.split_values[level] = split_value
        self.depth = max(self.depth, level + 1)
        
        # Update leaf values for the new level
        n_leaves = 2 ** self.depth
        if new_leaf_values.shape[0] == n_leaves:
            self.leaf_values = new_leaf_values
