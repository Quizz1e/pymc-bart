"""Metropolis-Hastings sampler for Decision Tables."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config

import pymc as pm
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, make_shared_replacements
from pytensor.tensor.variable import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.decision_table import DecisionTable, DecisionTableNode
from pymc_bart.split_rules import ContinuousSplitRule, SplitRule
from pymc_bart.utils import _encode_vi


class MHDecisionTableMove:
    """Base class for Decision Table moves."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
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
        rng : np.random.Generator
            Random number generator

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
        rng: np.random.Generator,
    ) -> tuple[DecisionTable, float]:
        """Propose growing a random leaf node."""
        new_table = table.copy()
        leaf_nodes = new_table.get_leaf_nodes(with_depth=True)

        if not leaf_nodes:
            return new_table, -np.inf

        # Select random leaf node
        leaf_idx = rng.integers(0, len(leaf_nodes))
        leaf_node, depth = leaf_nodes[leaf_idx]

        node_mask = _get_node_mask(new_table, leaf_node, X)
        if node_mask is None or not np.any(node_mask):
            return new_table, -np.inf

        split_var, split_value = new_table.get_level_predicate(depth)
        if split_var is None or split_value is None:
            split_var = rng.integers(0, X.shape[1])
            available_splits = _get_available_splits(X, split_var, node_mask)
            if available_splits.size == 0:
                return new_table, -np.inf

            split_value_raw = table.split_rules[split_var].get_split_value(available_splits)
            if split_value_raw is None:
                return new_table, -np.inf
            split_value = _ensure_split_array(split_value_raw)
        else:
            split_value = split_value.copy()

        split_rule = table.split_rules[split_var]
        division = _split_decision(split_rule, X[:, split_var], split_value)

        left_mask = node_mask & division
        right_mask = node_mask & (~division)

        if not left_mask.any() or not right_mask.any():
            return new_table, -np.inf

        left_value = _draw_leaf_value(Y, leaf_sd, left_mask, rng)
        right_value = _draw_leaf_value(Y, leaf_sd, right_mask, rng)

        # Grow the leaf
        new_table.grow_leaf_node(
            leaf_node=leaf_node,
            selected_predictor=split_var,
            split_value=split_value,
            left_value=left_value,
            right_value=right_value,
            left_nvalue=int(left_mask.sum()),
            right_nvalue=int(right_mask.sum()),
            depth=depth,
        )

        # Compute Hastings ratio
        n_leaf_nodes = len(leaf_nodes)
        n_split_nodes = new_table.count_split_nodes()

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
        rng: np.random.Generator,
    ) -> tuple[DecisionTable, float]:
        """Propose pruning a random split node."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = new_table.get_split_nodes(with_depth=True)

        if not split_nodes:
            return new_table, -np.inf

        n_split_nodes_before = len(split_nodes)

        # Select random split node
        split_idx = rng.integers(0, len(split_nodes))
        node_to_prune, _ = split_nodes[split_idx]

        # Check if both children are leaves
        if not all(child.is_leaf_node() for child in node_to_prune.children.values()):
            return new_table, -np.inf

        node_mask = _get_node_mask(new_table, node_to_prune, X)
        if node_mask is None or not node_mask.any():
            return new_table, -np.inf

        # Draw new leaf value
        new_leaf_value = _draw_leaf_value(Y, leaf_sd, node_mask, rng)

        # Prune: convert split node to leaf
        new_table.prune_node(
            node=node_to_prune,
            new_value=new_leaf_value,
            nvalue=int(node_mask.sum()),
        )

        # Compute Hastings ratio (reverse grow selects among new leaves)
        n_leaf_nodes_after = new_table.count_leaf_nodes()
        if n_leaf_nodes_after <= 0 or n_split_nodes_before <= 0:
            return new_table, -np.inf

        log_alpha = np.log(n_leaf_nodes_after) - np.log(n_split_nodes_before)

        return new_table, log_alpha


class ChangeMove(MHDecisionTableMove):
    """Change move: modify split rule of an existing split node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
    ) -> tuple[DecisionTable, float]:
        """Propose changing a split variable or split value."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = new_table.get_split_nodes(with_depth=True)

        if not split_nodes:
            return new_table, -np.inf

        # Select random split node
        split_idx = rng.integers(0, len(split_nodes))
        node, depth = split_nodes[split_idx]

        node_mask = _get_node_mask(new_table, node, X)
        if node_mask is None or not node_mask.any():
            return new_table, -np.inf

        # Change split variable (with some probability keep the same)
        if rng.random() < 0.5:
            new_split_var = node.idx_split_variable
        else:
            new_split_var = rng.integers(0, X.shape[1])

        # Get available split values for new variable
        available_splits = _get_available_splits(X, new_split_var, node_mask)
        if available_splits.size == 0:
            return new_table, -np.inf

        # Select split value
        split_value_raw = table.split_rules[new_split_var].get_split_value(available_splits)
        if split_value_raw is None:
            return new_table, -np.inf
        split_value = _ensure_split_array(split_value_raw)

        split_rule = table.split_rules[new_split_var]
        division = _split_decision(split_rule, X[:, new_split_var], split_value)
        left_mask = node_mask & division
        right_mask = node_mask & (~division)

        if not left_mask.any() or not right_mask.any():
            return new_table, -np.inf

        # Update node + depth predicate
        new_table.update_level_predicate(
            depth=depth,
            split_variable=new_split_var,
            split_value=split_value,
        )

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
    n_jobs : int
        Number of threads to evaluate tables in parallel (>=1). Defaults to 1.
    rng_seed : Optional[int]
        Seed used to initialize the sampler RNG. Defaults to None.
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
        n_jobs: int = 1,
        rng_seed: int | None = None,
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

        self.X = np.asarray(self.X, dtype=config.floatX)
        self.Y = np.asarray(self.Y, dtype=config.floatX)

        self.m = num_tables
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.leaf_sd = leaf_sd

        # Normalize move probabilities
        move_probs = np.array(move_probs)
        if np.any(move_probs <= 0):
            raise ValueError("move_probs must all be positive.")
        self.move_probs = move_probs / move_probs.sum()

        # Initialize move operators
        self.moves = [GrowMove(), PruneMove(), ChangeMove()]
        self.move_names = ["grow", "prune", "change"]
        self.reverse_move_idx = [1, 0, 2]
        self.rng = np.random.default_rng(rng_seed)
        self.n_jobs = max(1, int(n_jobs))

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

        self.table_predictions = [t.predict(self.X) for t in self.tables]
        self._y_ll = self.Y.astype(np.float64, copy=False).ravel()

        self.all_tables: list[list[DecisionTable]] = []
        self.accept_count = 0
        self.iteration = 0
        self.model = model
        self.tuning = True

        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.value_bart = value_bart

        super().__init__([value_bart], shared, **kwargs)

    def astep(self, _):
        """Execute one MH step."""
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        accept_rates: list[float] = []

        seeds = self.rng.integers(
            low=0,
            high=np.iinfo(np.int64).max,
            size=self.m,
            dtype=np.int64,
        )
        tasks = [
            (idx, self.tables[idx], self.table_predictions[idx], int(seeds[idx]))
            for idx in range(self.m)
        ]

        if self.n_jobs == 1:
            results = [self._run_single_step(*task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._run_single_step, *task) for task in tasks]
                results = [future.result() for future in futures]

        results.sort(key=lambda res: res["idx"])

        for result in results:
            idx = result["idx"]
            self.tables[idx] = result["table"]
            self.table_predictions[idx] = result["prediction"]
            self.accept_count += int(result["accepted"])
            accept_rates.append(float(result["accepted"]))
            if result["count_iteration"]:
                for var in result["split_vars"]:
                    variable_inclusion[var] += 1

        self.iteration += sum(1 for res in results if res["count_iteration"])

        if not self.tuning:
            self.all_tables.append([t.trim() for t in self.tables])

        # Compute ensemble predictions
        ensemble_pred = np.mean(np.stack(self.table_predictions, axis=0), axis=0)

        accept_rate = np.mean(accept_rates) if accept_rates else 0.0
        variable_inclusion_encoded = _encode_vi(variable_inclusion.tolist())
        last_move_idx = results[-1]["move_idx"] if results else 0

        stats = {
            "variable_inclusion": variable_inclusion_encoded,
            "move_type": self.move_names[last_move_idx],
            "accept_rate": accept_rate,
        }

        return ensemble_pred, [stats]

    def stop_tuning(self) -> None:
        """Mark the end of the tuning period and reset stored samples."""
        super().stop_tuning()
        self.tuning = False
        self.all_tables = []

    def _run_single_step(
        self,
        table_idx: int,
        table: DecisionTable,
        current_prediction: npt.NDArray,
        rng_seed: int,
    ) -> dict:
        """Execute a single MH proposal for one table (optionally in parallel)."""
        rng = np.random.default_rng(rng_seed)
        move_idx = rng.choice(len(self.moves), p=self.move_probs)
        move = self.moves[move_idx]
        reverse_idx = self.reverse_move_idx[move_idx]

        proposed_table, log_hastings = move.propose(
            table,
            self.X,
            self.Y,
            self.leaf_sd,
            rng,
        )

        if log_hastings == -np.inf:
            return {
                "idx": table_idx,
                "table": table,
                "prediction": current_prediction,
                "accepted": 0,
                "move_idx": move_idx,
                "split_vars": [],
                "count_iteration": False,
            }

        new_prediction = proposed_table.predict(self.X)
        log_likelihood_ratio = self._compute_log_likelihood_ratio(
            current_prediction, new_prediction
        )

        log_move_ratio = np.log(self.move_probs[reverse_idx]) - np.log(
            self.move_probs[move_idx]
        )
        log_alpha = log_likelihood_ratio + log_hastings + log_move_ratio
        accepted = int(np.log(rng.random()) < log_alpha)

        final_table = proposed_table if accepted else table
        final_prediction = new_prediction if accepted else current_prediction
        split_vars = self._get_split_variables(final_table)

        return {
            "idx": table_idx,
            "table": final_table,
            "prediction": final_prediction,
            "accepted": accepted,
            "move_idx": move_idx,
            "split_vars": split_vars,
            "count_iteration": True,
        }

    def _compute_log_likelihood_ratio(
        self,
        old_pred: npt.NDArray,
        new_pred: npt.NDArray,
    ) -> float:
        """Compute log likelihood ratio for MH acceptance."""
        old_flat = np.asarray(old_pred, dtype=np.float64).ravel()
        new_flat = np.asarray(new_pred, dtype=np.float64).ravel()

        if old_flat.shape[0] != self._y_ll.shape[0] or new_flat.shape[0] != self._y_ll.shape[0]:
            raise ValueError(
                "Predictions and observations must share the same flattened size."
            )

        return _log_likelihood_ratio_numba(
            self._y_ll,
            old_flat,
            new_flat,
            float(self.leaf_sd),
        )

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


def _get_available_splits(
    X: npt.NDArray, var_idx: int, mask: npt.NDArray | None = None
) -> npt.NDArray:
    """Get available split values for a variable."""
    values = X[:, var_idx]
    if mask is not None:
        mask = _normalize_mask(mask, values.shape[0])
        values = values[mask]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return values
    return np.unique(values)


def _draw_leaf_value(
    Y: npt.NDArray,
    leaf_sd: float,
    mask: npt.NDArray | None,
    rng: np.random.Generator,
) -> npt.NDArray:
    """Draw a leaf value from normal distribution."""
    if mask is not None and mask.any():
        mask = _normalize_mask(mask, Y.shape[0])
        target = Y[mask]
    else:
        target = Y
    return np.array([np.mean(target) + rng.normal(0.0, leaf_sd)])


def _get_node_mask(
    table: DecisionTable, target_node: DecisionTableNode, X: npt.NDArray
) -> npt.NDArray | None:
    """Return boolean mask of observations reaching the provided node."""
    split_rules = table.split_rules
    n_obs = X.shape[0]

    def _traverse(node: DecisionTableNode, mask: npt.NDArray) -> npt.NDArray | None:
        mask = _normalize_mask(mask, n_obs)
        if node is target_node:
            return mask
        if node.is_leaf_node():
            return None

        split_var = node.idx_split_variable
        split_value = node.value
        division = _split_decision(split_rules[split_var], X[:, split_var], split_value)

        left_mask = mask & division
        right_mask = mask & (~division)

        if 0 in node.children:
            result = _traverse(node.children[0], left_mask)
            if result is not None:
                return result
        if 1 in node.children:
            result = _traverse(node.children[1], right_mask)
            if result is not None:
                return result
        return None

    full_mask = np.ones(n_obs, dtype=bool)
    result = _traverse(table.root, full_mask)
    if result is None:
        return None
    return _normalize_mask(result, n_obs)


def _ensure_split_array(value) -> npt.NDArray:
    """Ensure split values are stored as numpy arrays."""
    if isinstance(value, np.ndarray):
        return value.copy()
    arr = np.array(value, copy=True)
    if arr.ndim == 0:
        arr = arr[None]
    return arr


def _normalize_mask(mask: npt.NDArray, length: int) -> npt.NDArray:
    """Ensure mask is 1-D boolean array of requested length."""
    mask_arr = np.asarray(mask, dtype=bool)
    mask_arr = np.squeeze(mask_arr)
    mask_arr = mask_arr.reshape(-1)
    if mask_arr.size != length:
        raise ValueError(
            f"Mask has size {mask_arr.size}, expected {length}. "
            "Split rule produced incompatible shape."
        )
    return mask_arr


def _split_decision(
    split_rule: SplitRule, feature_values: npt.NDArray, split_value: npt.NDArray
) -> npt.NDArray:
    """Evaluate split rule and normalize mask shape."""
    division = split_rule.divide(feature_values, split_value)
    return _normalize_mask(division, feature_values.shape[0])


@njit(cache=True, fastmath=True)
def _log_likelihood_ratio_numba(
    y: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    leaf_sd: float,
) -> float:
    """Numba-accelerated log-likelihood ratio."""
    inv_var = 1.0 / (leaf_sd * leaf_sd)
    sse_old = 0.0
    sse_new = 0.0
    for i in range(y.size):
        diff_old = y[i] - old_pred[i]
        diff_new = y[i] - new_pred[i]
        sse_old += diff_old * diff_old
        sse_new += diff_new * diff_new
    return 0.5 * (sse_old - sse_new) * inv_var


@dataclass
class BatchDiagnostics:
    """Summary statistics for one batch of MH updates."""

    batch_index: int
    draws_completed: int
    ensemble_mean: float
    ensemble_std: float
    accept_rate: float
    converged: bool
    diagnostics: dict


class ConvergenceController:
    """Heuristic controller that decides when to stop batched sampling."""

    def __init__(
        self,
        window: int = 5,
        tolerance: float = 1e-3,
        min_accept_rate: float = 0.02,
        max_rhat: float = 1.05,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.tolerance = tolerance
        self.min_accept_rate = min_accept_rate
        self.max_rhat = max_rhat
        self._summary_buffer: deque[float] = deque(maxlen=window)
        self._accept_buffer: deque[float] = deque(maxlen=window)

    def update(self, summary_stat: float, accept_rate: float) -> tuple[bool, dict]:
        """Update controller state and report convergence decision."""
        self._summary_buffer.append(float(summary_stat))
        self._accept_buffer.append(float(accept_rate))

        diagnostics = {
            "window": len(self._summary_buffer),
            "mean_prediction": float(np.mean(self._summary_buffer)),
            "mean_accept_rate": float(np.mean(self._accept_buffer)),
            "min_accept_rate": float(min(self._accept_buffer)),
            "delta": np.inf,
            "rhat": np.inf,
        }

        if len(self._summary_buffer) < self.window:
            return False, diagnostics

        mean_summary = diagnostics["mean_prediction"]
        latest = self._summary_buffer[-1]
        base = np.maximum(np.abs(mean_summary), 1e-12)
        diagnostics["delta"] = float(np.abs(latest - mean_summary) / base)
        diagnostics["rhat"] = float(self._estimate_rhat(np.asarray(self._summary_buffer)))

        converged = (
            diagnostics["delta"] <= self.tolerance
            and diagnostics["min_accept_rate"] >= self.min_accept_rate
            and diagnostics["rhat"] <= self.max_rhat
        )

        return bool(converged), diagnostics

    def _estimate_rhat(self, values: np.ndarray) -> float:
        """Approximate R-hat by splitting the window into two pseudo-chains."""
        if values.size < 4:
            return float("inf")

        half = values.size // 2
        first = values[:half]
        second = values[-half:]
        if first.size < 2:
            return float("inf")

        n = first.size
        mean_first = float(np.mean(first))
        mean_second = float(np.mean(second))
        mean_overall = float(np.mean(values))

        var_first = float(np.var(first, ddof=1))
        var_second = float(np.var(second, ddof=1))
        W = 0.5 * (var_first + var_second)
        if W <= 0:
            return 1.0

        B = n * ((mean_first - mean_overall) ** 2 + (mean_second - mean_overall) ** 2)
        var_hat = ((n - 1) / n) * W + (B / n)
        return float(np.sqrt(np.maximum(var_hat / W, 1.0)))


def run_batched_sampling(
    step: "MHDecisionTableSampler",
    draws_per_batch: int = 200,
    max_batches: int = 50,
    controller: ConvergenceController | None = None,
) -> tuple[list[BatchDiagnostics], ConvergenceController]:
    """
    Run MH sampler in batches until convergence controller signals stop.

    Parameters
    ----------
    step : MHDecisionTableSampler
        Initialized sampler instance (typically created inside a PyMC model).
    draws_per_batch : int
        Number of MH updates per batch before evaluating diagnostics.
    max_batches : int
        Maximum number of batches to execute.
    controller : Optional[ConvergenceController]
        Controller that evaluates convergence. Defaults to ConvergenceController().

    Returns
    -------
    tuple
        (diagnostics_history, controller)

    Notes
    -----
    This helper executes the sampler outside of PyMC's driver in order to
    support dynamic stopping. When used together with a PyMC model, ensure that
    other step methods are coordinated accordingly.
    """
    if controller is None:
        controller = ConvergenceController()

    diagnostics_history: list[BatchDiagnostics] = []
    total_draws = 0

    if getattr(step, "tuning", False):
        step.stop_tuning()

    for batch_idx in range(max_batches):
        batch_accept_rates = []
        ensemble_pred = None

        for _ in range(draws_per_batch):
            ensemble_pred, stats = step.astep(None)
            batch_accept_rates.append(stats[0]["accept_rate"])

        total_draws += draws_per_batch
        accept_rate = float(np.mean(batch_accept_rates)) if batch_accept_rates else 0.0

        if ensemble_pred is None:
            raise RuntimeError("Sampler did not produce any predictions in this batch.")

        summary_stat = float(np.mean(ensemble_pred))
        converged, diag = controller.update(summary_stat, accept_rate)

        batch_diag = BatchDiagnostics(
            batch_index=batch_idx,
            draws_completed=total_draws,
            ensemble_mean=summary_stat,
            ensemble_std=float(np.std(ensemble_pred)),
            accept_rate=accept_rate,
            converged=converged,
            diagnostics=diag,
        )
        diagnostics_history.append(batch_diag)

        if converged:
            break

    return diagnostics_history, controller
