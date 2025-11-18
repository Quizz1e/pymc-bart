import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from numba import njit
from pymc.initial_point import PointType
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config
from pytensor import function as pytensor_function
from pytensor.tensor.variable import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.split_rules import ContinuousSplitRule
from pymc_bart.decision_table import DecisionTable
from pymc_bart.utils import _encode_vi


class DecisionTableSampler(ArrayStepShared):
    """
    Metropolis-Hastings sampler for Decision Table BART.
    """
    
    name = "decision_table_bart"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes = {
        "variable_inclusion": (object, []),
        "tune": (bool, []),
    }
    
    def __init__(
        self,
        vars: list[pm.Distribution] | None = None,
        max_depth: int = 5,
        prob_grow: float = 0.25,
        prob_prune: float = 0.25,
        prob_change: float = 0.5,
        model: Model | None = None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        model = modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        
        if vars is None:
            raise ValueError("Unable to find variables to sample")
        
        # Filter to only BART variables
        bart_vars = []
        for var in vars:
            rv = model.values_to_rvs.get(var)
            if rv is not None and isinstance(rv.owner.op, BARTRV):
                bart_vars.append(var)
        
        if not bart_vars:
            raise ValueError("No BART variables found in the provided variables")
        
        if len(bart_vars) > 1:
            raise ValueError(
                "DecisionTableSampler can only handle one BART variable at a time."
            )
        
        value_bart = bart_vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op
        
        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X
        
        if isinstance(self.bart.Y, Variable):
            self.Y = self.bart.Y.eval()
        else:
            self.Y = self.bart.Y
        
        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.response = self.bart.response
        
        shape = initial_point[value_bart.name].shape
        self.shape = 1 if len(shape) == 1 else shape[0]
        
        # Decision table parameters
        self.max_depth = max_depth
        self.prob_grow = prob_grow
        self.prob_prune = prob_prune
        self.prob_change = prob_change
        
        # Initialize decision tables
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1
        
        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior
        
        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = [ContinuousSplitRule] * self.X.shape[1]
        
        # Initialize decision tables
        init_mean = self.Y.mean()
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))
        
        # Initialize leaf standard deviation
        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))
        y_unique = np.unique(self.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.Y.std() / self.m**0.5
        
        self.sum_trees = np.full(
            (self.trees_shape, self.leaves_shape, self.Y.shape[0]), init_mean
        ).astype(config.floatX)
        
        # Create initial decision tables
        self.all_tables = []
        for odim in range(self.trees_shape):
            odim_tables = []
            for _ in range(self.m):
                table = DecisionTable.new_table(
                    leaf_node_value=init_mean / self.m,
                    num_observations=self.num_observations,
                    shape=self.leaves_shape,
                    split_rules=self.split_rules,
                    max_depth=self.max_depth,
                )
                odim_tables.append(table)
            self.all_tables.append(odim_tables)
        
        self.tune = True
        self.iter = 0
        
        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.likelihood_logp = logp(initial_point, [model.datalogp], [value_bart], shared)
        
        super().__init__([value_bart], shared)
    
    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        
        for odim in range(self.trees_shape):
            for table_id in range(self.m):
                self.iter += 1
                
                # Compute sum of trees without current table
                current_table = self.all_tables[odim][table_id]
                current_pred = current_table._predict(self.X)
                self.sum_trees[odim] = self.sum_trees[odim] - current_pred
                
                # Propose new table
                proposed_table = self._propose_table(current_table, odim, table_id)
                proposed_pred = proposed_table._predict(self.X)
                
                # Metropolis-Hastings acceptance
                current_ll = self._compute_likelihood(current_pred, odim)
                proposed_ll = self._compute_likelihood(proposed_pred, odim)
                
                # Prior ratio (symmetric for simplicity)
                log_prior_ratio = 0.0
                
                # Proposal ratio (symmetric for simplicity)  
                log_proposal_ratio = 0.0
                
                log_accept_ratio = (proposed_ll - current_ll + 
                                  log_prior_ratio + log_proposal_ratio)
                
                if np.log(np.random.random()) < log_accept_ratio:
                    # Accept proposal
                    self.all_tables[odim][table_id] = proposed_table
                    self.sum_trees[odim] = self.sum_trees[odim] + proposed_pred
                    
                    # Update variable inclusion
                    for var in proposed_table.get_split_variables():
                        variable_inclusion[var] += 1
                else:
                    # Reject proposal - revert sum_trees
                    self.sum_trees[odim] = self.sum_trees[odim] + current_pred
        
        if not self.tune:
            # Convert tables to compatible format for BART
            table_arrays = np.array(self.all_tables)
            self.bart.all_trees.append(table_arrays)
        
        variable_inclusion = _encode_vi(variable_inclusion)
        stats = {"variable_inclusion": variable_inclusion, "tune": self.tune}
        
        return self.sum_trees, [stats]
    
    def _propose_table(self, current_table: DecisionTable, odim: int, table_id: int) -> DecisionTable:
        """Propose a new decision table."""
        proposed_table = current_table.copy()
        
        move_type = np.random.choice(
            ['grow', 'prune', 'change'], 
            p=[self.prob_grow, self.prob_prune, self.prob_change]
        )
        
        if move_type == 'grow' and proposed_table.depth < self.max_depth:
            self._propose_grow(proposed_table, odim)
        elif move_type == 'prune' and proposed_table.depth > 0:
            self._propose_prune(proposed_table)
        elif move_type == 'change' and proposed_table.depth > 0:
            self._propose_change(proposed_table, odim)
        
        return proposed_table
    
    def _propose_grow(self, table: DecisionTable, odim: int) -> None:
        """Propose growing the table by adding a new level."""
        level_to_grow = table.depth  # Grow at the next level
        
        # Sample splitting variable
        split_var = self._sample_splitting_variable()
        
        # Sample splitting value
        split_val = self._sample_splitting_value(split_var)
        
        # Sample new leaf values
        new_leaf_values = self._sample_leaf_values(table, level_to_grow, odim)
        
        table.grow_level(level_to_grow, split_var, split_val, new_leaf_values)
    
    def _propose_prune(self, table: DecisionTable) -> None:
        """Propose pruning the table by removing the last level."""
        if table.depth == 0:
            return
            
        level_to_prune = table.depth - 1
        
        # Set the level to inactive
        table.split_variables[level_to_prune] = -1
        table.depth = level_to_prune
        
        # Update leaf values (average child leaf values)
        n_leaves = 2 ** table.depth
        new_leaf_values = np.zeros((n_leaves, table.leaf_values.shape[1]))
        
        for i in range(n_leaves):
            left_child = table.leaf_values[2 * i]
            right_child = table.leaf_values[2 * i + 1]
            new_leaf_values[i] = (left_child + right_child) / 2
        
        table.leaf_values = new_leaf_values
    
    def _propose_change(self, table: DecisionTable, odim: int) -> None:
        """Propose changing a split at a random level."""
        if table.depth == 0:
            return
            
        level_to_change = np.random.randint(0, table.depth)
        
        # Sample new splitting variable and value
        split_var = self._sample_splitting_variable()
        split_val = self._sample_splitting_value(split_var)
        
        table.split_variables[level_to_change] = split_var
        table.split_values[level_to_change] = split_val
        
        # Resample leaf values
        new_leaf_values = self._sample_leaf_values(table, table.depth, odim)
        table.leaf_values = new_leaf_values
    
    def _sample_splitting_variable(self) -> int:
        """Sample a splitting variable."""
        probs = self.alpha_vec / self.alpha_vec.sum()
        return np.random.choice(len(probs), p=probs)
    
    def _sample_splitting_value(self, split_var: int) -> float:
        """Sample a splitting value for the given variable."""
        values = self.X[:, split_var]
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return 0.0
            
        return np.random.choice(valid_values)
    
    def _sample_leaf_values(self, table: DecisionTable, depth: int, odim: int) -> npt.NDArray:
        """Sample new leaf values for the table."""
        n_leaves = 2 ** depth
        leaf_values = np.zeros((n_leaves, self.leaves_shape))
        
        for i in range(n_leaves):
            # Find data points that would reach this leaf
            leaf_mask = self._get_leaf_mask(table, i)
            
            if np.sum(leaf_mask) > 0:
                # Sample from posterior of leaf value
                residuals = self.Y[leaf_mask] - self.sum_trees[odim, :, leaf_mask].mean(axis=0)
                mean_val = residuals.mean() if len(residuals) > 0 else 0
                leaf_values[i] = np.random.normal(mean_val, self.leaf_sd[odim])
            else:
                leaf_values[i] = np.random.normal(0, self.leaf_sd[odim])
        
        return leaf_values
    
    def _get_leaf_mask(self, table: DecisionTable, leaf_idx: int) -> npt.NDArray[np.bool_]:
        """Get boolean mask for data points that reach the given leaf."""
        mask = np.ones(self.X.shape[0], dtype=bool)
        
        # Reconstruct the path to the leaf
        path = []
        temp_idx = leaf_idx
        for level in range(table.depth - 1, -1, -1):
            if temp_idx % 2 == 0:  # Left child
                path.append(('left', level))
            else:  # Right child
                path.append(('right', level))
            temp_idx = temp_idx // 2
        path.reverse()
        
        # Apply splits along the path
        for direction, level in path:
            split_var = table.split_variables[level]
            split_val = table.split_values[level]
            
            if split_var == -1:
                continue
                
            rule = table.split_rules[split_var]
            if direction == 'left':
                split_mask = rule.divide(self.X[:, split_var], split_val)
            else:
                split_mask = ~rule.divide(self.X[:, split_var], split_val)
            
            mask = mask & split_mask
        
        return mask
    
    def _compute_likelihood(self, prediction: npt.NDArray, odim: int) -> float:
        """Compute log-likelihood for given prediction."""
        delta = (
            np.identity(self.trees_shape)[odim][:, None, None]
            * prediction[None, :, :]
        )
        
        return self.likelihood_logp((self.sum_trees + delta).flatten())
    
    @staticmethod
    def competence(var: pm.Distribution, has_grad: bool) -> Competence:
        """DecisionTableSampler is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
    
    @staticmethod
    def _make_update_stats_functions():
        def update_stats(step_stats):
            return {key: step_stats[key] for key in ("variable_inclusion", "tune")}
        
        return (update_stats,)


def logp(
    point,
    out_vars: list[pm.Distribution],
    vars: list[pm.Distribution],
    shared: list[pt.TensorVariable],
):
    """Compile PyTensor function of the model and the input and output variables."""
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    function = pytensor_function([inarray0], out_list[0])
    function.trust_input = True
    return function
