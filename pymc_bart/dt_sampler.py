# dt_sampler.py
#   Metropolis-Hastings sampler for Decision Table BART (decision tables)
#   Implements moves: change_threshold, change_feature, add_depth, remove_depth,
#   change_leaf_values (no swap).

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from .decision_table import DecisionTable
from .split_rules import ContinuousSplitRule, SplitRule

__all__ = ["DTMHSampler", "default_move_probs"]


# Default move probabilities (must sum to 1.0)
default_move_probs = {
    "change_threshold": 0.30,
    "change_feature": 0.25,
    "add_depth": 0.10,
    "remove_depth": 0.05,
    "change_leaf_values": 0.30,
}


@dataclass
class SamplerStats:
    attempts: Dict[str, int]
    accepts: Dict[str, int]

    def __init__(self, moves: List[str]):
        self.attempts = {m: 0 for m in moves}
        self.accepts = {m: 0 for m in moves}

    def record(self, move: str, accepted: bool):
        self.attempts[move] += 1
        if accepted:
            self.accepts[move] += 1

    def acceptance_rates(self) -> Dict[str, float]:
        rates = {}
        for m in self.attempts:
            a = self.attempts[m]
            rates[m] = self.accepts[m] / a if a > 0 else 0.0
        return rates


class DTMHSampler:
    """
    Metropolis-Hastings sampler for an ensemble of decision-table trees.

    This sampler operates on an ensemble of `m` decision tables (DecisionTable instances).
    Each step proposes changes to one tree at a time using a chosen move from
    {change_threshold, change_feature, add_depth, remove_depth, change_leaf_values}.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix (n_obs, n_features)
    Y : np.ndarray
        Response vector (n_obs,)
    m : int
        Number of trees in the ensemble
    init_depth : int
        Initial depth for each decision table
    split_rules : Optional[list[SplitRule]]
        List of SplitRule objects, one per feature (if None -> ContinuousSplitRule for each feature)
    split_prior : Optional[np.ndarray]
        Prior weights for features (length = n_features). If None, uniform.
    sigma : float
        Observation noise std (assumed known/fixed). You may sample externally and pass value.
    tau : float
        Prior std for leaf values (Gaussian prior N(0, tau^2)).
    depth_penalty : float
        Linear penalty on depth in log-prior: log_prior_depth = -depth_penalty * depth
        (simple prior; replaceable by Chipman-style prior).
    rng : Optional[np.random.Generator]
        Random generator.
    move_probs : Optional[dict]
        Mapping move_name -> probability (must sum to 1).
    """

    def __init__(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        m: int = 20,
        init_depth: int = 1,
        split_rules: Optional[List[SplitRule]] = None,
        split_prior: Optional[npt.NDArray] = None,
        sigma: float = 1.0,
        tau: float = 1.0,
        depth_penalty: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        move_probs: Optional[Dict[str, float]] = None,
        max_depth: int = 6,
    ) -> None:
        self.X = np.asarray(X)
        self.Y = np.asarray(Y).ravel()
        self.n_obs, self.n_features = self.X.shape
        self.m = int(m)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.depth_penalty = float(depth_penalty)
        self.max_depth = int(max_depth)

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        if split_rules is None:
            self.split_rules = [ContinuousSplitRule()] * self.n_features
        else:
            self.split_rules = split_rules

        if split_prior is None:
            self.split_prior = np.ones(self.n_features) / self.n_features
        else:
            sp = np.asarray(split_prior, dtype=float)
            sp = sp / sp.sum()
            self.split_prior = sp

        self.move_probs = default_move_probs.copy() if move_probs is None else move_probs.copy()
        # normalize move_probs
        s = sum(self.move_probs.values())
        if abs(s - 1.0) > 1e-12:
            for k in self.move_probs:
                self.move_probs[k] /= s

        self._moves = list(self.move_probs.keys())
        self.stats = SamplerStats(self._moves)

        # Initialize m identical shallow decision tables (depth = init_depth)
        self.trees: List[DecisionTable] = []
        leaf_init = np.array([0.0])
        for _ in range(self.m):
            dt = DecisionTable.new_table(
                depth=init_depth,
                leaf_node_value=leaf_init,
                num_observations=self.n_obs,
                shape=1,
                split_rules=self.split_rules,
            )
            # set random level predicates initially using split_rules and split_prior
            for lvl in range(init_depth):
                feat = int(self.rng.choice(self.n_features, p=self.split_prior))
                # choose split value uniformly from observed values of that feature
                vals = np.unique(self.X[:, feat])
                if vals.size == 0:
                    thr = 0.0
                else:
                    thr = float(vals[self.rng.integers(0, vals.size)])
                dt.set_level_predicate(lvl, feat, thr)
            # initial routing & leaf value assignment (posterior mean w.r.t. residual = Y / m)
            dt.route_data(self.X)
            # initialize leaf values to residual/m (simple)
            avg = self.Y.mean() / float(self.m)
            dt.set_leaf_values([np.array([avg])] * len(dt.idx_leaf_nodes))
            self.trees.append(dt)

        # maintain ensemble sum (vector length n_obs)
        self.sum_trees = np.zeros(self.n_obs, dtype=float)
        for t in self.trees:
            pred = self._pred_vector(t)
            self.sum_trees += pred

    # -------------------------
    # Helpers
    # -------------------------
    def _pred_vector(self, tree: DecisionTable) -> npt.NDArray:
        """Return tree prediction as 1D array (n_obs,)."""
        p = tree._predict()  # returns (shape, n_obs) as in Tree._predict
        # assume shape==1 for now
        if p.ndim == 2 and p.shape[0] == 1:
            return p[0].astype(float)
        # flatten general case: sum across shape dims
        return np.sum(p, axis=0).astype(float)

    def _log_prior_structure(self, tree: DecisionTable) -> float:
        """
        Simple log-prior for decision table:
        - depth penalty: -depth_penalty * depth
        - feature prior: sum log(split_prior[feature_at_level]) over levels
        """
        lp = -self.depth_penalty * float(tree.depth)
        # feature prior: if some level has idx -1 (unset) treat uniform
        for lvl in range(tree.depth):
            feat, _ = tree.get_level_predicate(lvl)
            if feat is None or feat < 0:
                lp += -np.log(self.n_features)
            else:
                lp += np.log(self.split_prior[int(feat)] + 1e-12)
        return float(lp)

    def _marginal_loglik_tree(self, tree: DecisionTable, residual: npt.NDArray) -> float:
        """
        Marginal log-likelihood of residuals under tree, integrating out leaf parameter theta
        with prior N(0, tau^2) and Gaussian noise sigma^2.

        For each leaf with observations y_i:
            p(y_leaf) = (2πσ^2)^{-(n-1)/2} * (2π(σ^2/n + τ^2))^{-1/2} *
                        exp( -0.5 * sum((y - ybar)^2)/σ^2 - 0.5*(ybar - 0)^2 / (σ^2/n + τ^2) )
        We'll compute log of that and sum across leaves.
        """
        sigma2 = self.sigma ** 2
        tau2 = self.tau ** 2
        log_marg = 0.0
        for idx in tree.idx_leaf_nodes:
            node = tree.get_node(idx)
            obs_idx = node.idx_data_points
            if obs_idx is None or obs_idx.size == 0:
                # no data -> predictive contribution is constant; ignore (0)
                continue
            y = residual[obs_idx]
            n = y.size
            # sum squared deviations around leaf mean
            ybar = float(np.mean(y))
            ss_within = float(np.sum((y - ybar) ** 2))  # scale part
            # first factor: contribution from residuals around ybar
            if n > 1:
                term1 = -0.5 * (n - 1) * np.log(2 * np.pi * sigma2)
            else:
                term1 = 0.0  # no within-sum term
            term2 = -0.5 * ss_within / sigma2
            # marginal part for ybar given prior:
            var_ybar = sigma2 / n
            marginal_var = var_ybar + tau2
            term3 = -0.5 * np.log(2 * np.pi * marginal_var) - 0.5 * ((ybar - 0.0) ** 2) / marginal_var
            log_marg += term1 + term2 + term3
        return float(log_marg)

    # -------------------------
    # Move proposals
    # -------------------------
    def _propose_change_threshold(self, tree: DecisionTable) -> Tuple[DecisionTable, Dict]:
        """
        Propose new split value at a randomly chosen level.
        Proposal: choose level uniformly among 0..depth-1; choose new threshold randomly
        among unique X[:,feat] values for that feature.
        """
        new_tree = tree.copy()
        level = int(self.rng.integers(0, tree.depth))
        feat, _ = tree.get_level_predicate(level)
        # if feat unset, draw feature first
        if feat is None or feat < 0:
            feat = int(self.rng.choice(self.n_features, p=self.split_prior))
        # pick new thr uniformly among unique observed values for that feature
        vals = np.unique(self.X[:, feat])
        if vals.size == 0:
            thr = 0.0
        else:
            thr = float(vals[self.rng.integers(0, vals.size)])
        new_tree.set_level_predicate(level, int(feat), thr)
        new_tree.route_data(self.X)
        info = {"move": "change_threshold", "level": level, "old_feature": feat}
        return new_tree, info

    def _propose_change_feature(self, tree: DecisionTable) -> Tuple[DecisionTable, Dict]:
        """
        Propose change of feature at a randomly chosen level.
        Proposal: pick level uniformly, propose new feature according to split_prior (excluding current)
        and choose threshold uniformly from unique values of new feature.
        """
        new_tree = tree.copy()
        level = int(self.rng.integers(0, tree.depth))
        old_feat, old_thr = tree.get_level_predicate(level)
        # sample new feature proportional to split_prior but exclude current
        probs = self.split_prior.copy()
        if old_feat is not None and old_feat >= 0:
            probs[int(old_feat)] = 0.0
            probs = probs / probs.sum()
        new_feat = int(self.rng.choice(self.n_features, p=probs))
        vals = np.unique(self.X[:, new_feat])
        if vals.size == 0:
            new_thr = 0.0
        else:
            new_thr = float(vals[self.rng.integers(0, vals.size)])
        new_tree.set_level_predicate(level, new_feat, new_thr)
        new_tree.route_data(self.X)
        info = {"move": "change_feature", "level": level, "old_feature": old_feat}
        return new_tree, info

    def _propose_add_depth(self, tree: DecisionTable) -> Tuple[DecisionTable, Dict]:
        """
        Propose to increase depth by 1 (if under max_depth).
        New level predicates are sampled from split_prior and observed values.
        Implementation: build a new DecisionTable with depth+1, copy existing level predicates, sample
        new level's (feature, thr) and route data.
        """
        if tree.depth >= self.max_depth:
            return tree.copy(), {"move": "add_depth", "allowed": False}
        new_depth = tree.depth + 1
        # collect current level predicates
        feat_list = []
        thr_list = []
        for lvl in range(tree.depth):
            f, t = tree.get_level_predicate(lvl)
            if f is None:
                f = int(self.rng.choice(self.n_features, p=self.split_prior))
            if t is None:
                vals_tmp = np.unique(self.X[:, int(f)])
                t = float(vals_tmp[self.rng.integers(0, vals_tmp.size)]) if vals_tmp.size > 0 else 0.0
            feat_list.append(int(f))
            thr_list.append(float(t))
        # sample new level pred
        new_feat = int(self.rng.choice(self.n_features, p=self.split_prior))
        vals = np.unique(self.X[:, new_feat])
        new_thr = float(vals[self.rng.integers(0, vals.size)]) if vals.size > 0 else 0.0
        feat_list.append(new_feat)
        thr_list.append(new_thr)
        # create new table
        leaf_init = np.array([0.0])
        new_tree = DecisionTable.new_table(
            depth=new_depth,
            leaf_node_value=leaf_init,
            num_observations=self.n_obs,
            shape=1,
            split_rules=self.split_rules,
            feature_per_level=feat_list,
            split_value_per_level=thr_list,
        )
        new_tree.route_data(self.X)
        info = {"move": "add_depth", "old_depth": tree.depth}
        return new_tree, info

    def _propose_remove_depth(self, tree: DecisionTable) -> Tuple[DecisionTable, Dict]:
        """
        Propose to decrease depth by 1 (if depth > 1).
        Create new DecisionTable with depth-1, copy level predicates 0..depth-2.
        """
        if tree.depth <= 1:
            return tree.copy(), {"move": "remove_depth", "allowed": False}
        new_depth = tree.depth - 1
        feat_list = []
        thr_list = []
        for lvl in range(new_depth):
            f, t = tree.get_level_predicate(lvl)
            if f is None:
                f = int(self.rng.choice(self.n_features, p=self.split_prior))
            if t is None:
                vals_tmp = np.unique(self.X[:, int(f)])
                t = float(vals_tmp[self.rng.integers(0, vals_tmp.size)]) if vals_tmp.size > 0 else 0.0
            feat_list.append(int(f))
            thr_list.append(float(t))
        leaf_init = np.array([0.0])
        new_tree = DecisionTable.new_table(
            depth=new_depth,
            leaf_node_value=leaf_init,
            num_observations=self.n_obs,
            shape=1,
            split_rules=self.split_rules,
            feature_per_level=feat_list,
            split_value_per_level=thr_list,
        )
        new_tree.route_data(self.X)
        info = {"move": "remove_depth", "old_depth": tree.depth}
        return new_tree, info

    def _propose_change_leaf_values(self, tree: DecisionTable, prop_scale: float = 0.1) -> Tuple[DecisionTable, Dict]:
        """
        Propose local random-walk on leaf values.
        v' = v + eps, eps ~ N(0, prop_scale^2)
        Symmetric proposal => q ratio cancels.
        """
        new_tree = tree.copy()
        new_values = []
        for idx in new_tree.idx_leaf_nodes:
            v = np.asarray(new_tree.get_node(idx).value).reshape(-1)
            # scalar case
            eps = self.rng.normal(0.0, prop_scale, size=v.shape)
            new_v = (v + eps).reshape(-1)
            new_tree.get_node(idx).value = np.array(new_v)
            new_values.append(np.array(new_v))
        # routing not necessary (structure unchanged)
        info = {"move": "change_leaf_values", "prop_scale": prop_scale}
        return new_tree, info

    # -------------------------
    # MH step for a single tree
    # -------------------------
    def _mh_step_one_tree(self, tree_idx: int) -> None:
        """
        Propose a move for tree `tree_idx` and accept/reject under MH.
        We use marginal likelihood (integrating leaf theta) for structural moves,
        and full-likelihood (theta present) for leaf-value moves.
        """
        # pick move at random by move_probs
        moves = list(self.move_probs.keys())
        probs = np.array([self.move_probs[m] for m in moves], dtype=float)
        move_choice = self.rng.choice(moves, p=probs)

        self.stats.attempts[move_choice] += 1

        old_tree = self.trees[tree_idx]
        old_pred = self._pred_vector(old_tree)
        sum_no_t = self.sum_trees - old_pred
        # residual for tree (data assigned to this tree): Y - sum_no_t
        residual = self.Y - sum_no_t

        # Propose new tree depending on move
        if move_choice == "change_threshold":
            new_tree, info = self._propose_change_threshold(old_tree)
            # compute marginal loglik for old and new
            marg_old = self._marginal_loglik_tree(old_tree, residual)
            marg_new = self._marginal_loglik_tree(new_tree, residual)
            lp_old = self._log_prior_structure(old_tree)
            lp_new = self._log_prior_structure(new_tree)
            # no q ratio for symmetric proposals (we chose thr uniformly from obs values on feature)
            log_q_ratio = 0.0
            log_accept_ratio = (marg_new + lp_new) - (marg_old + lp_old) + log_q_ratio
            accepted = False
            if np.log(self.rng.random()) < log_accept_ratio:
                # accept: set tree's leaf values to posterior means for prediction
                self._set_tree_leaf_posterior_mean(new_tree, residual)
                self.trees[tree_idx] = new_tree
                # update sum_trees
                new_pred = self._pred_vector(new_tree)
                self.sum_trees = sum_no_t + new_pred
                accepted = True

        elif move_choice == "change_feature":
            new_tree, info = self._propose_change_feature(old_tree)
            marg_old = self._marginal_loglik_tree(old_tree, residual)
            marg_new = self._marginal_loglik_tree(new_tree, residual)
            lp_old = self._log_prior_structure(old_tree)
            lp_new = self._log_prior_structure(new_tree)
            log_q_ratio = 0.0
            log_accept_ratio = (marg_new + lp_new) - (marg_old + lp_old) + log_q_ratio
            accepted = False
            if np.log(self.rng.random()) < log_accept_ratio:
                self._set_tree_leaf_posterior_mean(new_tree, residual)
                self.trees[tree_idx] = new_tree
                new_pred = self._pred_vector(new_tree)
                self.sum_trees = sum_no_t + new_pred
                accepted = True

        elif move_choice == "add_depth":
            new_tree, info = self._propose_add_depth(old_tree)
            if info.get("allowed", True) is False:
                accepted = False
            else:
                # proposal asymmetry: q(remove | new) vs q(add | old)
                # We approximate q(remove|new)=move_probs['remove_depth'] (if allowed) and q(add|old)=move_probs['add_depth']
                # More exact q could depend on number of levels etc.
                q_forward = self.move_probs.get("add_depth", 1e-12)
                q_backward = self.move_probs.get("remove_depth", 1e-12)
                marg_old = self._marginal_loglik_tree(old_tree, residual)
                marg_new = self._marginal_loglik_tree(new_tree, residual)
                lp_old = self._log_prior_structure(old_tree)
                lp_new = self._log_prior_structure(new_tree)
                log_q_ratio = np.log(q_backward + 1e-300) - np.log(q_forward + 1e-300)
                log_accept_ratio = (marg_new + lp_new) - (marg_old + lp_old) + log_q_ratio
                accepted = False
                if np.log(self.rng.random()) < log_accept_ratio:
                    self._set_tree_leaf_posterior_mean(new_tree, residual)
                    self.trees[tree_idx] = new_tree
                    new_pred = self._pred_vector(new_tree)
                    self.sum_trees = sum_no_t + new_pred
                    accepted = True

        elif move_choice == "remove_depth":
            new_tree, info = self._propose_remove_depth(old_tree)
            if info.get("allowed", True) is False:
                accepted = False
            else:
                q_forward = self.move_probs.get("remove_depth", 1e-12)
                q_backward = self.move_probs.get("add_depth", 1e-12)
                marg_old = self._marginal_loglik_tree(old_tree, residual)
                marg_new = self._marginal_loglik_tree(new_tree, residual)
                lp_old = self._log_prior_structure(old_tree)
                lp_new = self._log_prior_structure(new_tree)
                log_q_ratio = np.log(q_backward + 1e-300) - np.log(q_forward + 1e-300)
                log_accept_ratio = (marg_new + lp_new) - (marg_old + lp_old) + log_q_ratio
                accepted = False
                if np.log(self.rng.random()) < log_accept_ratio:
                    self._set_tree_leaf_posterior_mean(new_tree, residual)
                    self.trees[tree_idx] = new_tree
                    new_pred = self._pred_vector(new_tree)
                    self.sum_trees = sum_no_t + new_pred
                    accepted = True

        elif move_choice == "change_leaf_values":
            # propose local gaussian perturbation of leaf values
            new_tree, info = self._propose_change_leaf_values(old_tree, prop_scale=0.1)
            # compute full log-likelihood: data given sum_no_t + tree_pred
            new_pred = self._pred_vector(new_tree)
            old_pred = old_pred
            sse_old = float(np.sum((self.Y - (sum_no_t + old_pred)) ** 2))
            sse_new = float(np.sum((self.Y - (sum_no_t + new_pred)) ** 2))
            # log likelihood diff
            ll_diff = -0.5 * (sse_new - sse_old) / (self.sigma ** 2)
            # prior on leaf values: Gaussian with mean 0 and var tau^2
            lp_old_leaves = self._log_prior_leaves(old_tree)
            lp_new_leaves = self._log_prior_leaves(new_tree)
            log_accept_ratio = ll_diff + (lp_new_leaves - lp_old_leaves)
            accepted = False
            if np.log(self.rng.random()) < log_accept_ratio:
                # accept: assign new leaf values
                self.trees[tree_idx] = new_tree
                self.sum_trees = sum_no_t + new_pred
                accepted = True

        else:
            # unknown move
            accepted = False

        # record stats
        self.stats.record(move_choice, accepted)

    def _log_prior_leaves(self, tree: DecisionTable) -> float:
        """Gaussian prior N(0, tau^2) on each leaf value (scalar case)."""
        lp = 0.0
        tau2 = self.tau ** 2
        for idx in tree.idx_leaf_nodes:
            v = np.asarray(tree.get_node(idx).value).reshape(-1)
            lp += np.sum(-0.5 * np.log(2 * np.pi * tau2) - 0.5 * (v ** 2) / tau2)
        return float(lp)

    def _set_tree_leaf_posterior_mean(self, tree: DecisionTable, residual: npt.NDArray) -> None:
        """
        Deterministically set leaf values to posterior mean given residual = Y - sum_other_trees.
        Posterior mean for leaf with observations y: mean_post = (sum(y)/σ^2) / (n/σ^2 + 1/τ^2)
        (assuming prior mean = 0).
        """
        sigma2 = self.sigma ** 2
        tau2 = self.tau ** 2
        for idx in tree.idx_leaf_nodes:
            node = tree.get_node(idx)
            obs_idx = node.idx_data_points
            if obs_idx is None or obs_idx.size == 0:
                post_mean = 0.0
            else:
                y = residual[obs_idx]
                n = y.size
                denom = (n / sigma2) + (1.0 / tau2)
                post_mean = (np.sum(y) / sigma2) / denom
            node.value = np.array([float(post_mean)])

    # -------------------------
    # Public interface: run sampler
    # -------------------------
    def run(
        self,
        n_iter: int = 1000,
        burn: int = 0,
        thin: int = 1,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, List]:
        """
        Run the MH sampler.

        Returns dictionary `trace` containing recorded sum_trees (after burn & thinning)
        and acceptance statistics.

        trace = {
           "sum_trees": [array(n_obs,), ...],
           "acceptance_rates": {move: rate, ...}
        }
        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        saved = []
        total_iters = int(n_iter)
        record_every = int(thin)

        for it in range(total_iters):
            # sweep over trees
            for t_idx in range(self.m):
                self._mh_step_one_tree(t_idx)

            # after full sweep record if beyond burn and matches thinning
            if it >= burn and ((it - burn) % record_every == 0):
                # store a copy
                saved.append(np.array(self.sum_trees.copy()))
            if verbose and (it % max(1, total_iters // 10) == 0):
                print(f"[DTMHSampler] iter {it}/{total_iters}")

        trace = {"sum_trees": saved, "acceptance_rates": self.stats.acceptance_rates()}
        return trace
