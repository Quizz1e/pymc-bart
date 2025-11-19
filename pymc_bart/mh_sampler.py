"""Metropolis-Hastings sampler for Decision Tables."""

import numpy as np
import numpy.typing as npt
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config

import pymc as pm
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, make_shared_replacements
from pytensor.tensor.variable import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.decision_table import DecisionTable, DecisionTableNode
from pymc_bart.split_rules import ContinuousSplitRule
from pymc_bart.utils import _encode_vi


class MHDecisionTableMove:
    """Base class for Decision Table moves."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
    ) -> tuple[DecisionTable, float]:
        """
        Propose a new tree structure.

        Parameters
        ----------
        table : DecisionTable
            Current decision table
        X : npt.NDArray
            Input data
        Y : npt.NDArray
            Response variable
        leaf_sd : float
            Standard deviation for leaf values

        Returns
        -------
        tuple[DecisionTable, float]
            New table and log Hastings ratio
        """
        raise NotImplementedError


class GrowMove(MHDecisionTableMove):
    """Grow move: expand a leaf node into a split node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
    ) -> tuple[DecisionTable, float]:
        """Propose growing a random leaf node."""
        new_table = table.copy()
        leaf_nodes = new_table.get_leaf_nodes()

        if not leaf_nodes:
            return new_table, -np.inf

        # Select random leaf node
        leaf_idx = np.random.randint(0, len(leaf_nodes))
        leaf_node = leaf_nodes[leaf_idx]

        # Select random split variable
        split_var = np.random.randint(0, X.shape[1])

        # Get available split values
        available_splits = _get_available_splits(X, split_var)
        if available_splits.size == 0:
            return new_table, -np.inf

        # Select random split value
        split_value = table.split_rules[split_var].get_split_value(available_splits)
        if split_value is None:
            return new_table, -np.inf

        # Compute new leaf values
        to_left = table.split_rules[split_var].divide(available_splits, split_value).astype(bool)
        left_value = _draw_leaf_value(Y, leaf_sd)
        right_value = _draw_leaf_value(Y, leaf_sd)

        # Grow the leaf
        new_table.grow_leaf_node(
            leaf_node=leaf_node,
            selected_predictor=split_var,
            split_value=np.array([split_value]),
            left_value=left_value,
            right_value=right_value,
            left_nvalue=np.sum(to_left),
            right_nvalue=np.sum(~to_left),
        )

        # Compute Hastings ratio
        n_leaf_nodes = len(leaf_nodes)
        n_split_nodes = len(new_table.get_leaf_nodes()) - 1

        log_alpha = np.log(max(n_split_nodes, 1)) - np.log(n_leaf_nodes)

        return new_table, log_alpha


class PruneMove(MHDecisionTableMove):
    """Prune move: collapse a split node into a leaf node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
    ) -> tuple[DecisionTable, float]:
        """Propose pruning a random split node."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = [
            node
            for node in _get_all_nodes(new_table.root)
            if node.is_split_node()
        ]

        if not split_nodes:
            return new_table, -np.inf

        # Select random split node
        split_idx = np.random.randint(0, len(split_nodes))
        node_to_prune = split_nodes[split_idx]

        # Check if both children are leaves
        if not all(child.is_leaf_node() for child in node_to_prune.children.values()):
            return new_table, -np.inf

        # Draw new leaf value
        new_leaf_value = _draw_leaf_value(Y, leaf_sd)

        # Prune: convert split node to leaf
        node_to_prune.idx_split_variable = -1
        node_to_prune.value = new_leaf_value
        node_to_prune.children = {}

        # Compute Hastings ratio
        leaf_nodes = new_table.get_leaf_nodes()
        n_leaf_nodes = len(leaf_nodes)
        n_split_nodes = len(split_nodes) - 1

        log_alpha = np.log(n_leaf_nodes) - np.log(max(n_split_nodes, 1))

        return new_table, log_alpha


class ChangeMove(MHDecisionTableMove):
    """Change move: modify split rule of an existing split node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
    ) -> tuple[DecisionTable, float]:
        """Propose changing a split variable or split value."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = [
            node
            for node in _get_all_nodes(new_table.root)
            if node.is_split_node()
        ]

        if not split_nodes:
            return new_table, -np.inf

        # Select random split node
        split_idx = np.random.randint(0, len(split_nodes))
        node = split_nodes[split_idx]

        # Change split variable (with some probability keep the same)
        if np.random.random() < 0.5:
            new_split_var = node.idx_split_variable
        else:
            new_split_var = np.random.randint(0, X.shape[1])

        # Get available split values for new variable
        available_splits = _get_available_splits(X, new_split_var)
        if available_splits.size == 0:
            return new_table, -np.inf

        # Select split value
        split_value = table.split_rules[new_split_var].get_split_value(available_splits)
        if split_value is None:
            return new_table, -np.inf

        # Update node
        node.idx_split_variable = new_split_var
        node.value = np.array([split_value])

        # Hastings ratio = 1 (symmetric proposal)
        log_alpha = 0.0

        return new_table, log_alpha


class MHDecisionTableSampler(ArrayStepShared):
    """
    Metropolis-Hastings sampler for Decision Tables.

    Parameters
    ----------
    vars : list
        List of value variables for sampler
    num_tables : int
        Number of decision tables. Defaults to 50
    move_probs : tuple[float, float, float]
        Probabilities for (grow, prune, change) moves. Defaults to (0.33, 0.33, 0.34)
    leaf_sd : float
        Standard deviation for leaf values. Defaults to 1.0
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    initial_point : Optional dict
        Initial point for sampling
    """

    name = "mh_decision_table"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "variable_inclusion": (object, []),
        "move_type": (str, []),
        "accept_rate": (float, []),
    }

    def __init__(
        self,
        vars: list[pm.Distribution] | None = None,
        num_tables: int = 50,
        move_probs: tuple[float, float, float] = (0.33, 0.33, 0.34),
        leaf_sd: float = 1.0,
        model: Model | None = None,
        initial_point: dict | None = None,
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
                "MH sampler can only handle one BART variable at a time."
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

        self.m = num_tables
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.leaf_sd = leaf_sd

        # Normalize move probabilities
        move_probs = np.array(move_probs)
        self.move_probs = move_probs / move_probs.sum()

        # Initialize move operators
        self.moves = [GrowMove(), PruneMove(), ChangeMove()]
        self.move_names = ["grow", "prune", "change"]

        # Initialize decision tables
        self.tables = [
            DecisionTable.new_decision_table(
                leaf_node_value=np.array([self.Y.mean() / self.m]),
                num_observations=self.num_observations,
                shape=1,
                split_rules=self.bart.split_rules
                if self.bart.split_rules
                else [ContinuousSplitRule] * self.num_variates,
            )
            for _ in range(self.m)
        ]

        self.all_tables = [[t.trim() for t in self.tables]]
        self.accept_count = 0
        self.iteration = 0
        self.model = model

        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.value_bart = value_bart

        super().__init__([value_bart], shared, **kwargs)

    def astep(self, _):
        """Execute one MH step."""
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        accept_rates = []
        move_idx = 0

        for table_idx in range(self.m):
            # Select move type
            move_idx = np.random.choice(len(self.moves), p=self.move_probs)
            move = self.moves[move_idx]

            # Propose new table
            proposed_table, log_hastings = move.propose(
                self.tables[table_idx], self.X, self.Y, self.leaf_sd
            )

            if log_hastings == -np.inf:
                accept_rates.append(0.0)
                continue

            # Compute log likelihood ratio
            old_pred = self.tables[table_idx].predict(self.X)
            new_pred = proposed_table.predict(self.X)

            log_likelihood_ratio = self._compute_log_likelihood_ratio(old_pred, new_pred)

            # Acceptance probability
            log_alpha = log_likelihood_ratio + log_hastings
            if np.log(np.random.random()) < log_alpha:
                self.tables[table_idx] = proposed_table
                self.accept_count += 1
                accept_rates.append(1.0)
            else:
                accept_rates.append(0.0)

            self.iteration += 1

            # Track variable inclusion
            split_vars = self._get_split_variables(self.tables[table_idx])
            for var in split_vars:
                variable_inclusion[var] += 1

        # Store all tables for posterior inference
        self.all_tables.append([t.trim() for t in self.tables])

        # Compute ensemble predictions
        ensemble_pred = np.mean(
            np.array([t.predict(self.X) for t in self.tables]), axis=0
        )

        accept_rate = np.mean(accept_rates) if accept_rates else 0.0
        variable_inclusion_encoded = _encode_vi(variable_inclusion.tolist())

        stats = {
            "variable_inclusion": variable_inclusion_encoded,
            "move_type": self.move_names[move_idx],
            "accept_rate": accept_rate,
        }

        return ensemble_pred, [stats]

    def _compute_log_likelihood_ratio(
        self,
        old_pred: npt.NDArray,
        new_pred: npt.NDArray,
    ) -> float:
        """Compute log likelihood ratio for MH acceptance."""
        residuals_old = self.Y - old_pred
        residuals_new = self.Y - new_pred

        sse_old = np.sum(residuals_old**2)
        sse_new = np.sum(residuals_new**2)

        log_lik_ratio = 0.5 * (sse_old - sse_new) / (self.leaf_sd**2)

        return log_lik_ratio

    def _get_split_variables(self, table: DecisionTable) -> list[int]:
        """Get all split variables used in the table."""
        split_vars = []

        def _traverse(node: DecisionTableNode):
            if node.is_split_node():
                split_vars.append(node.idx_split_variable)
                for child in node.children.values():
                    _traverse(child)

        _traverse(table.root)
        return split_vars

    @staticmethod
    def competence(var: pm.Distribution, has_grad: bool) -> Competence:
        """MH sampler is suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    @staticmethod
    def _make_update_stats_functions():
        def update_stats(step_stats):
            return {
                key: step_stats[key]
                for key in ("variable_inclusion", "move_type", "accept_rate")
            }

        return (update_stats,)


def _get_available_splits(X: npt.NDArray, var_idx: int) -> npt.NDArray:
    """Get available split values for a variable."""
    values = X[:, var_idx]
    return values[~np.isnan(values)]


def _draw_leaf_value(Y: npt.NDArray, leaf_sd: float) -> npt.NDArray:
    """Draw a leaf value from normal distribution."""
    return np.array([np.mean(Y) + np.random.normal(0, leaf_sd)])


def _get_all_nodes(node: DecisionTableNode) -> list[DecisionTableNode]:
    """Get all nodes in the tree."""
    nodes = [node]
    for child in node.children.values():
        nodes.extend(_get_all_nodes(child))
    return nodes
