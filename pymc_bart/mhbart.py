# mhbart.py
# Metropolis-Hastings BART с Decision Tables (без Particle Gibbs)

import numpy as np
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.model import modelcontext
from pymc.pytensorf import make_shared_replacements, join_nonshared_inputs
from pymc.initial_point import PointType
from .decision_table import DecisionTable
from .tree import RunningSd  # используем тот же RunningSd
from .pgbart import (SampleSplittingVariable, NormalSampler, UniformSampler,
                     compute_prior_probability, draw_leaf_value, filter_missing_values,
                     fast_mean, _encode_vi, logp)

class MHBART(ArrayStepShared):
    name = "mhbart"
    generates_stats = True
    stats_dtypes_shapes = {"variable_inclusion": (object, []), "tune": (bool, [])}

    def __init__(self, vars=None, batch=(0.1, 0.1), model=None, initial_point: PointType | None = None, **kwargs):
        model = modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()
        if vars is None:
            vars = model.value_vars
        bart_var = [v for v in vars if v.owner.op.name == "BART"][0]
        self.bart = model.values_to_rvs[bart_var].owner.op

        self.X = self.bart.X if not isinstance(self.bart.X, Variable) else self.bart.X.eval()
        self.Y = self.bart.Y if not isinstance(self.bart.Y, Variable) else self.bart.Y.eval()
        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.response = self.bart.response
        self.shape = bart_var.shape[0] if bart_var.ndim > 1 else 1
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        self.alpha_vec = np.ones(self.X.shape[1]) if self.bart.split_prior.size == 0 else self.bart.split_prior.copy()
        self.split_rules = [ContinuousSplitRule] * self.X.shape[1] if self.bart.split_rules is None else self.bart.split_rules
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        init_mean = self.Y.mean()
        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape)) * (self.Y.std() / self.m**0.5)
        if len(np.unique(self.Y)) == 2:
            self.leaf_sd *= 3 / self.m**0.5

        self.running_sd = [RunningSd((self.leaves_shape, self.num_observations)) for _ in range(self.trees_shape)]

        self.sum_trees = np.full((self.trees_shape, self.leaves_shape, self.num_observations), init_mean, dtype=config.floatX)
        self.normal = NormalSampler(1, self.leaves_shape)
        self.ssv = SampleSplittingVariable(self.alpha_vec)
        self.prior_prob_leaf_node = compute_prior_probability(self.bart.alpha, self.bart.beta)

        self.tune = True
        batch_0 = max(1, int(self.m * batch[0]))
        batch_1 = max(1, int(self.m * batch[1]))
        self.batch = (batch_0, batch_1)
        self.lower = 0

        shared = make_shared_replacements(initial_point, [bart_var], model)
        self.likelihood_logp = logp(initial_point, [model.datalogp], [bart_var], shared)

        self.all_trees = [[DecisionTable.new_tree(init_mean / self.m, np.arange(self.num_observations),
                                                  self.num_observations, self.leaves_shape, self.split_rules)
                           for _ in range(self.m)] for _ in range(self.trees_shape)]

        for odim in range(self.trees_shape):
            for tree in self.all_trees[odim]:
                self.sum_trees[odim] += tree._predict()

        super().__init__([bart_var], shared)

    def compute_log_prior(self, tree):
        d = len(tree.levels)
        log_p = 0.0
        for k in range(d):
            p_split = self.bart.alpha * (1 + k) ** (-self.bart.beta)
            log_p += np.log(p_split)
            feature = tree.levels[k][0]
            p_feature = self.alpha_vec[feature] / self.alpha_vec.sum()
            log_p += np.log(p_feature)
        p_stop = 1 - self.bart.alpha * (1 + d) ** (-self.bart.beta)
        log_p += np.log(p_stop)
        return log_p

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        upper = min(self.lower + self.batch[not self.tune], self.m)
        tree_ids = range(self.lower, upper)
        self.lower = upper if upper < self.m else 0

        for odim in range(self.trees_shape):
            for tree_id in tree_ids:
                current_tree = self.all_trees[odim][tree_id]
                sum_trees_noi = self.sum_trees[odim] - current_tree._predict()

                new_tree, delta_logp, prior_ratio, prop_ratio = self.propose(odim, tree_id, current_tree, sum_trees_noi)

                if new_tree is None:   # proposal rejected internally
                    continue

                accept_prob = np.exp(delta_logp + prior_ratio + prop_ratio)
                accept_prob = min(1.0, accept_prob)

                if np.random.random() < accept_prob:
                    # accept
                    self.all_trees[odim][tree_id] = new_tree
                    self.sum_trees[odim] = sum_trees_noi + new_tree._predict()
                    current_tree = new_tree
                    if self.tune:
                        for f in new_tree.get_split_variables():
                            self.alpha_vec[f] += 1
                        self.ssv = SampleSplittingVariable(self.alpha_vec)
                        self.running_sd[odim].update(new_tree._predict())
                    else:
                        for f in new_tree.get_split_variables():
                            variable_inclusion[f] += 1

        if not self.tune:
            self.bart.all_trees.append(np.array([[t.trim() for t in trees] for trees in self.all_trees]))

        stats = {"variable_inclusion": _encode_vi(variable_inclusion), "tune": self.tune}
        return self.sum_trees, [stats]

    def propose(self, odim, tree_id, current_tree, sum_trees_noi):
        depth = len(current_tree.levels)
        if depth == 0:
            move = "grow"
        else:
            ps = np.array([0.25, 0.25, 0.5]) if depth > 1 else np.array([0.5, 0.5, 0.0])
            ps = ps / ps.sum()
            move = np.random.choice(["grow", "prune", "change"], p=ps)

        old_prior = self.compute_log_prior(current_tree)
        old_predict = current_tree._predict()
        old_logp = self.likelihood_logp(self.sum_trees[odim].flatten())

        if move == "grow":
            success, new_tree, prop_forward, prop_backward = self.grow_move(odim, current_tree, sum_trees_noi)
            if not success:
                return None, 0, 0, 0
        elif move == "prune":
            if depth == 0:
                return None, 0, 0, 0
            new_tree, prop_forward, prop_backward = self.prune_move(odim, current_tree, sum_trees_noi)
        else:  # change
            if depth == 0:
                return None, 0, 0, 0
            level = np.random.randint(depth)
            new_tree, prop_forward, prop_backward = self.change_move(odim, current_tree, sum_trees_noi, level)

        new_predict = new_tree._predict()
        new_full = sum_trees_noi + new_predict
        new_logp = self.likelihood_logp(new_full.flatten())
        delta_logp = new_logp - old_logp
        new_prior = self.compute_log_prior(new_tree)
        prior_ratio = new_prior - old_prior
        prop_ratio = np.log(prop_backward / prop_forward) if prop_forward > 0 else -np.inf

        return new_tree, delta_logp, prior_ratio, prop_ratio

    def grow_move(self, odim, tree, sum_trees_noi):
        feature = self.ssv.rvs()
        selected = self.available_predictors[feature]
        avail, _ = filter_missing_values(self.X[:, selected], np.arange(self.num_observations), self.missing_data)
        if avail.size <= 1:
            return False, None, 0, 0
        split_idx = int(np.random.random() * avail.size)
        split_value = avail[split_idx]
        p_split_value = 1.0 / avail.size
        p_feature = self.alpha_vec[feature] / self.alpha_vec.sum()
        prop_forward = 0.25 * p_feature * p_split_value   # приблизительно, точнее подгоняется в astep

        # создаём новую таблицу
        new_levels = tree.levels + [(selected, split_value)]
        num_new = len(tree.leaf_values) * 2
        new_leaf_values = np.zeros((num_new, self.leaves_shape), dtype=config.floatX)
        new_leaf_nvalues = np.zeros(num_new, dtype=int)
        new_idx_per_leaf = []

        grew = False
        for old in range(len(tree.leaf_values)):
            idxs = tree.idx_data_points_per_leaf[old]
            if len(idxs) == 0:
                left = right = np.array([], dtype=int)
            else:
                vals = self.X[idxs, selected]
                mask = ~np.isnan(vals)
                vals = vals[mask]
                idxs = idxs[mask]
                to_left = self.split_rules[selected].divide(vals, split_value)
                left = idxs[to_left]
                right = idxs[~to_left]
                if len(left) > 0 and len(right) > 0:
                    grew = True
            left_idx = old * 2
            right_idx = old * 2 + 1
            new_idx_per_leaf.append(left)
            new_idx_per_leaf.append(right)
            new_leaf_nvalues[left_idx] = len(left)
            new_leaf_nvalues[right_idx] = len(right)

            for side, side_idxs in enumerate([left, right]):
                leaf_idx = old * 2 + side
                ypred = sum_trees_noi[side_idxs] if len(side_idxs) > 0 else np.array([0.0])
                value, _ = draw_leaf_value(ypred, None, self.m,
                                           self.normal.rvs() * self.leaf_sd[odim],
                                           self.leaves_shape, self.response)
                new_leaf_values[leaf_idx] = value

        if not grew:
            return False, None, 0, 0

        new_output = np.zeros_like(tree.output)
        new_tree = DecisionTable(new_levels, new_leaf_values, new_leaf_nvalues,
                                 new_idx_per_leaf, tree.split_rules, new_output, self.leaves_shape)

        prop_backward = 0.25   # prune
        return True, new_tree, prop_forward, prop_backward

    def prune_move(self, odim, tree, sum_trees_noi):
        if len(tree.levels) == 0:
            return None, 0, 0

        last_feature, last_split = tree.levels[-1]
        p_feature = self.alpha_vec[last_feature] / self.alpha_vec.sum()
        avail, _ = filter_missing_values(self.X[:, last_feature], np.arange(self.num_observations), self.missing_data)
        p_split_value = 1.0 / avail.size if avail.size > 0 else 0.0
        prop_forward = 0.25
        prop_backward = 0.25 * p_feature * p_split_value

        new_levels = tree.levels[:-1]
        num_new = len(tree.leaf_values) // 2
        new_leaf_values = np.zeros((num_new, self.leaves_shape), dtype=config.floatX)
        new_idx_per_leaf = []
        new_leaf_nvalues = np.zeros(num_new, dtype=int)

        for new_i in range(num_new):
            left = tree.idx_data_points_per_leaf[new_i*2]
            right = tree.idx_data_points_per_leaf[new_i*2+1]
            merged = np.concatenate((left, right))
            new_idx_per_leaf.append(merged)
            new_leaf_nvalues[new_i] = len(merged)
            ypred = sum_trees_noi[merged] if len(merged) > 0 else np.array([0.0])
            value, _ = draw_leaf_value(ypred, None, self.m,
                                       self.normal.rvs() * self.leaf_sd[odim],
                                       self.leaves_shape, self.response)
            new_leaf_values[new_i] = value

        new_output = np.zeros_like(tree.output)
        new_tree = DecisionTable(new_levels, new_leaf_values, new_leaf_nvalues,
                                 new_idx_per_leaf, tree.split_rules, new_output, self.leaves_shape)

        return new_tree, prop_forward, prop_backward

    def change_move(self, odim, tree, sum_trees_noi, level):
        old_feature, old_split = tree.levels[level]
        p_feature_old = self.alpha_vec[old_feature] / self.alpha_vec.sum()
        avail_old, _ = filter_missing_values(self.X[:, old_feature], np.arange(self.num_observations), self.missing_data)
        p_split_old = 1.0 / avail_old.size if avail_old.size > 0 else 0.0

        feature = self.ssv.rvs()
        selected = self.available_predictors[feature]
        avail, _ = filter_missing_values(self.X[:, selected], np.arange(self.num_observations), self.missing_data)
        if avail.size <= 1:
            return tree, 0, 0
        split_idx = int(np.random.random() * avail.size)
        split_value = avail[split_idx]

        p_feature_new = self.alpha_vec[feature] / self.alpha_vec.sum()
        p_split_new = 1.0 / avail.size

        prop_forward = 0.5 * (1.0 / len(tree.levels)) * p_feature_new * p_split_new
        prop_backward = 0.5 * (1.0 / len(tree.levels)) * p_feature_old * p_split_old

        new_tree = tree.copy()
        new_tree.levels[level] = (selected, split_value)
        new_tree.update_idx_data_points(self.X, self.missing_data)

        for leaf in range(len(new_tree.leaf_values)):
            idxs = new_tree.idx_data_points_per_leaf[leaf]
            ypred = sum_trees_noi[idxs] if len(idxs) > 0 else np.array([0.0])
            value, _ = draw_leaf_value(ypred, None, self.m,
                                       self.normal.rvs() * self.leaf_sd[odim],
                                       self.leaves_shape, self.response)
            new_tree.leaf_values[leaf] = value

        return new_tree, prop_forward, prop_backward

    @staticmethod
    def competence(var, has_grad):
        if var.owner.op.name == "BART":
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
