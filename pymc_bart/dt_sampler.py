# dt_sampler.py
#   Metropolis-Hastings sampler for Decision Table BART (decision tables)
#   Implements moves: change_threshold, change_feature, add_depth, remove_depth,
#   change_leaf_values (no swap).

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .decision_table import DecisionTable
from .split_rules import ContinuousSplitRule

__all__ = ["DTMHSampler"]


class DTMHSampler:
    """
    Metropolis-Hastings sampler for an ensemble of symmetric decision tables.

    Parameters
    ----------
    X : npt.NDArray
        Covariate matrix (n_obs, n_features).
    Y : npt.NDArray
        Response vector (n_obs,).
    m : int
        Number of trees in the ensemble.
    init_depth : int
        Initial depth for each decision table.
    split_rules : list[SplitRule] | None
        List of split rules; default ContinuousSplitRule for all features.
    sigma : float | None
        Noise standard deviation (if None, will be estimated from Y).
    tau : float
        Prior std for leaf values.
    alpha, beta : floats
        (Optional) hyperparams controlling prior on depth (we use a simple prior: -beta*log(1+depth)).
    rng : Optional[np.random.Generator]
        RNG for reproducibility.
    moves_prob : Optional[Dict[str, float]]
        Dictionary with probabilities for choosing moves.
        Keys: "change_threshold", "change_feature", "add_depth", "remove_depth", "change_leaf"
    max_depth : int
        Maximum permitted depth for trees (to keep state space bounded).
    """

    def __init__(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        m: int = 20,
        init_depth: int = 1,
        split_rules: Optional[List[Any]] = None,
        sigma: Optional[float] = None,
        tau: float = 1.0,
        alpha: float = 0.95,
        beta: float = 2.0,
        rng: Optional[np.random.Generator] = None,
        moves_prob: Optional[Dict[str, float]] = None,
        max_depth: int = 6,
    ) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y).astype(float)
        self.n, self.p = self.X.shape
        self.m = int(m)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_depth = int(max_depth)

        self.rng = rng if rng is not None else np.random.default_rng()

        if split_rules is None:
            self.split_rules = [ContinuousSplitRule] * self.p
        else:
            self.split_rules = split_rules

        if sigma is None:
            self.sigma = float(np.std(self.Y))
        else:
            self.sigma = float(sigma)

        default_moves = {
            "change_threshold": 0.4,
            "change_feature": 0.2,
            "add_depth": 0.1,
            "remove_depth": 0.1,
            "change_leaf": 0.2,
        }
        if moves_prob is None:
            self.moves_prob = default_moves
        else:
            mp = default_moves.copy()
            mp.update(moves_prob)
            total = sum(mp.values())
            self.moves_prob = {k: v / total for k, v in mp.items()}

        leaf_init = np.array([0.0])
        self.trees: List[DecisionTable] = []
        for _ in range(self.m):
            dt = DecisionTable.new_table(
                depth=init_depth,
                leaf_node_value=leaf_init,
                num_observations=self.n,
                shape=1,
                split_rules=[sr() if isinstance(sr, type) else sr for sr in self.split_rules],
                feature_per_level=None,
                split_value_per_level=None,
            )
            for level in range(init_depth):
                feat = int(self.rng.integers(0, self.p))
                thr = self._choose_split_value(feat)
                dt.set_level_predicate(level, feat, thr)
            dt.route_data(self.X)
            dt.draw_leaf_values_posterior(residual=self.Y - 0.0, sigma=self.sigma, tau=self.tau, rng=self.rng)
            self.trees.append(dt)

        self.sum_predictions = self._compute_ensemble_prediction()
        self.iteration = 0

    def _compute_ensemble_prediction(self) -> npt.NDArray:
        """Return vector (n,) equal to sum of predictions of all trees."""
        total = np.zeros(self.n, dtype=float)
        for t in self.trees:
            pred = t._predict() 
            if pred.ndim == 2:
                total += pred[0]
            else:
                total += pred
        return total

    def _choose_split_value(self, feature: int) -> float:
        """Choose a split value for feature: pick random observed value (possibly jittered)."""
        col = self.X[:, feature]
        unique_vals = np.unique(col[~np.isnan(col)])
        if unique_vals.size == 0:
            return 0.0
        val = unique_vals[self.rng.integers(0, unique_vals.size)]
        jitter = 1e-8 * (np.std(col) if np.std(col) > 0 else 1.0)
        return float(val + self.rng.normal(0, jitter))

    def _log_likelihood(self, ensemble_pred: npt.NDArray) -> float:
        """Gaussian likelihood log p(Y | f, sigma) up to additive constant."""
        resid = self.Y - ensemble_pred
        sse = float(np.sum(resid * resid))
        return -0.5 * sse / (self.sigma ** 2)

    def _log_prior_tree_structure(self, tree: DecisionTable) -> float:
        """
        Simple prior on tree depth: discourage large depth
        log p(depth) = - beta * log(1 + depth)
        (This is a simplification of Chipman-style prior; replace if needed.)
        Plus prior on leaf values (Gaussian with std=tau).
        """
        d = tree.depth
        lp_depth = -self.beta * np.log(1.0 + d)
        leaf_vals = tree.get_leaf_values()
        lp_leaf = 0.0
        for v in leaf_vals:
            arr = np.asarray(v).reshape(-1)
            lp_leaf += -0.5 * float(np.sum((arr ** 2) / (self.tau ** 2)))
        return float(lp_depth + lp_leaf)

    def _log_posterior(self, ensemble_pred: npt.NDArray, tree: DecisionTable) -> float:
        """Sum of log-likelihood and log-prior (up to additive constant)."""
        return self._log_likelihood(ensemble_pred) + self._log_prior_tree_structure(tree)

    def _select_move(self) -> str:
        """Randomly pick a move according to moves_prob."""
        keys = list(self.moves_prob.keys())
        probs = np.array([self.moves_prob[k] for k in keys], dtype=float)
        idx = self.rng.choice(len(keys), p=probs)
        return keys[idx]

    def _move_change_threshold(self, tree: DecisionTable) -> Optional[DecisionTable]:
        """Pick random level and add Gaussian perturbation to threshold (symmetric proposal)."""
        if tree.depth < 1:
            return None
        new_tree = tree.copy()
        level = int(self.rng.integers(0, tree.depth))
        feat, thr = new_tree.get_level_predicate(level)
        if feat is None:
            feat = int(self.rng.integers(0, self.p))
        proposal_sd = max(1e-2, np.std(self.X[:, feat]) * 0.05)
        new_thr = float(thr + self.rng.normal(0.0, proposal_sd))
        new_tree.set_level_predicate(level, feat, new_thr)
        new_tree.route_data(self.X)
        return new_tree

    def _move_change_feature(self, tree: DecisionTable) -> Optional[DecisionTable]:
        """Pick random level and change the feature used on that level (uniform among other features)."""
        if tree.depth < 1:
            return None
        new_tree = tree.copy()
        level = int(self.rng.integers(0, tree.depth))
        old_feat, old_thr = new_tree.get_level_predicate(level)
        if old_feat is None:
            old_feat = int(self.rng.integers(0, self.p))
        candidates = list(range(self.p))
        if len(candidates) <= 1:
            return None
        candidates.remove(int(old_feat))
        new_feat = int(self.rng.choice(candidates))
        new_thr = self._choose_split_value(new_feat)
        new_tree.set_level_predicate(level, new_feat, new_thr)
        new_tree.route_data(self.X)
        return new_tree

     def _move_add_depth(self, tree: DecisionTable) -> Optional[DecisionTable]:
        """Add one level (depth+1) if not exceeding max_depth."""
        if tree.depth >= self.max_depth:
            return None
        new_depth = tree.depth + 1
        feature_per_level = []
        split_value_per_level = []
        for lvl in range(tree.depth):
            f, s = tree.get_level_predicate(lvl)
            if f is None:
                f = int(self.rng.integers(0, self.p))
            if s is None:
                s = self._choose_split_value(int(f))
            feature_per_level.append(int(f))
            split_value_per_level.append(float(s))
        new_feat = int(self.rng.integers(0, self.p))
        new_thr = self._choose_split_value(new_feat)
        feature_per_level.append(new_feat)
        split_value_per_level.append(new_thr)
        leaf_init = np.array([0.0])
        split_rules_instances = [
            sr() if isinstance(sr, type) else sr.__class__() for sr in self.split_rules
        ]
        new_tree = DecisionTable.new_table(
            depth=new_depth,
            leaf_node_value=leaf_init,
            num_observations=self.n,
            shape=1,
            split_rules=split_rules_instances,
            feature_per_level=feature_per_level,
            split_value_per_level=split_value_per_level,
        )
        new_tree.route_data(self.X)
        return new_tree

    def _move_remove_depth(self, tree: DecisionTable) -> Optional[DecisionTable]:
        """Remove lowest level (depth-1) if depth > 1."""
        if tree.depth <= 1:
            return None
        new_depth = tree.depth - 1
        feature_per_level = []
        split_value_per_level = []
        for lvl in range(new_depth):
            f, s = tree.get_level_predicate(lvl)
            if f is None:
                f = int(self.rng.integers(0, self.p))
            if s is None:
                s = self._choose_split_value(int(f))
            feature_per_level.append(int(f))
            split_value_per_level.append(float(s))
        leaf_init = np.array([0.0])
        split_rules_instances = [
            sr() if isinstance(sr, type) else sr.__class__() for sr in self.split_rules
        ]
        new_tree = DecisionTable.new_table(
            depth=new_depth,
            leaf_node_value=leaf_init,
            num_observations=self.n,
            shape=1,
            split_rules=split_rules_instances,
            feature_per_level=feature_per_level,
            split_value_per_level=split_value_per_level,
        )
        new_tree.route_data(self.X)
        return new_tree

    def _move_change_leaf(self, tree: DecisionTable) -> Optional[DecisionTable]:
        """Propose local random-walk on leaf values (symmetric)."""
        new_tree = tree.copy()
        prop_sd = max(1e-3, self.tau * 0.1)
        leaf_vals = new_tree.get_leaf_values()
        new_vals = []
        for v in leaf_vals:
            arr = np.asarray(v).reshape(-1)
            new_arr = arr + self.rng.normal(0.0, prop_sd, size=arr.shape)
            new_vals.append(new_arr)
        new_tree.set_leaf_values(new_vals)
        new_tree.route_data(self.X)
        return new_tree

    def _attempt_update_tree(self, tree_idx: int, move: Optional[str] = None, draw_leaves_after_accept: bool = True) -> Tuple[bool, float]:
        """
        Try to update tree `tree_idx` using a randomly chosen or given move.
        Returns (accepted: bool, log_accept_ratio: float).
        """
        old_tree = self.trees[tree_idx]
        pred_old_arr = old_tree._predict()
        pred_old = pred_old_arr[0] if pred_old_arr.ndim == 2 else pred_old_arr
        sum_no_t = self.sum_predictions - pred_old

        if move is None:
            move = self._select_move()

        new_tree = None
        if move == "change_threshold":
            new_tree = self._move_change_threshold(old_tree)
        elif move == "change_feature":
            new_tree = self._move_change_feature(old_tree)
        elif move == "add_depth":
            new_tree = self._move_add_depth(old_tree)
        elif move == "remove_depth":
            new_tree = self._move_remove_depth(old_tree)
        elif move == "change_leaf":
            new_tree = self._move_change_leaf(old_tree)
        else:
            return False, 0.0

        if new_tree is None:
            return False, float("-inf")

        residual = self.Y - sum_no_t

        leaf_means = []
        for idx in new_tree.idx_leaf_nodes:
            node = new_tree.get_node(idx)
            obs_idx = node.idx_data_points
            if obs_idx is None or obs_idx.size == 0:
                mean_post = 0.0
            else:
                y_leaf = residual[obs_idx]
                n_leaf = y_leaf.size
                denom = (n_leaf / (self.sigma ** 2)) + (1.0 / (self.tau ** 2))
                var_post = 1.0 / denom
                mean_part = (np.sum(y_leaf) / (self.sigma ** 2)) + (0.0 / (self.tau ** 2))
                mean_post = var_post * mean_part
            leaf_means.append(np.array([mean_post]))
        new_tree.set_leaf_values(leaf_means)
        new_tree.route_data(self.X)

        pred_new_arr = new_tree._predict()
        pred_new = pred_new_arr[0] if pred_new_arr.ndim == 2 else pred_new_arr
        sum_new = sum_no_t + pred_new

        logpost_old = self._log_posterior(self.sum_predictions, old_tree)
        logpost_new = self._log_posterior(sum_new, new_tree)

        log_accept = logpost_new - logpost_old
        accept = False
        u = np.log(self.rng.random())
        if u < log_accept:
            accept = True
            if draw_leaves_after_accept:
                new_tree.draw_leaf_values_posterior(residual=self.Y - sum_no_t, sigma=self.sigma, tau=self.tau, rng=self.rng)
                new_tree.route_data(self.X)
                pred_new_arr = new_tree._predict()
                pred_new = pred_new_arr[0] if pred_new_arr.ndim == 2 else pred_new_arr
                sum_new = sum_no_t + pred_new
            self.trees[tree_idx] = new_tree
            self.sum_predictions = sum_new
        else:
            accept = False

        return accept, float(log_accept)

    def run(
        self,
        n_iter: int,
        burn_in: int = 0,
        thin: int = 1,
        store_trees: bool = False,
        draw_leaves_after_accept: bool = True,
    ) -> Dict[str, Any]:
        """
        Run sampler.

        Returns dictionary with:
            - 'preds': array (num_saved, n_obs) sum_predictions at saved iterations
            - 'accept_rate': overall acceptance rate
            - 'trees' (optional): list of stored tree ensembles (trimmed) at saved iterations if store_trees True
        """
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        saved_preds = []
        saved_trees = []
        total_attempts = 0
        total_accepts = 0

        for it in range(n_iter):
            self.iteration += 1
            for t_idx in range(self.m):
                total_attempts += 1
                accepted, log_acc = self._attempt_update_tree(t_idx, move=None, draw_leaves_after_accept=draw_leaves_after_accept)
                if accepted:
                    total_accepts += 1
            if it >= burn_in and ((it - burn_in) % thin == 0):
                saved_preds.append(np.array(self.sum_predictions))
                if store_trees:
                    trimmed = [tr.trim() for tr in self.trees]
                    saved_trees.append(trimmed)

        results = {"preds": np.asarray(saved_preds), "accept_rate": total_accepts / max(1, total_attempts)}
        if store_trees:
            results["trees"] = saved_trees
        return results
