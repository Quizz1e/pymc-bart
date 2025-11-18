# mhbart.py

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

import warnings
import numpy as np
import pytensor.tensor as pt
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pymc.initial_point import PointType
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pytensor import function as pytensor_function

from .bart import BARTRV
from .decision_table import DecisionTable
from .pgbart import RunningSd, SampleSplittingVariable, compute_prior_probability, NormalSampler, UniformSampler, logp, jitter_duplicated, _encode_vi
from .split_rules import ContinuousSplitRule

class MHBART(ArrayStepShared):
    """Metropolis-Hastings sampler for BART with decision tables (no Particle Gibbs)."""
    
    name = "mhbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes = {
        "variable_inclusion": (object, []),
        "tune": (bool, []),
        "accept": (float, []),
    }

    def __init__(
        self,
        vars=None,
        num_proposals=20,  # Number of MH proposals per tree update (for efficiency)
        batch=(0.1, 0.1),
        model=None,
        initial_point=None,
        compile_kwargs=None,
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

        bart_vars = []
        for var in vars:
            rv = model.values_to_rvs.get(var)
            if rv is not None and isinstance(rv.owner.op, BARTRV):
                bart_vars.append(var)

        if not bart_vars:
            raise ValueError("No BART variables found.")

        if len(bart_vars) > 1:
            raise ValueError("MHBART handles one BART variable at a time.")

        value_bart = bart_vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        self.X = self.bart.X.eval() if isinstance(self.bart.X, pt.TensorVariable) else self.bart.X
        self.Y = self.bart.Y.eval() if isinstance(self.bart.Y, pt.TensorVariable) else self.bart.Y

        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.response = self.bart.response

        shape = initial_point[value_bart.name].shape
        self.shape = 1 if len(shape) == 1 else shape[0]

        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        self.split_rules = self.bart.split_rules or [ContinuousSplitRule] * self.X.shape[1]

        for idx, rule in enumerate(self.split_rules):
            if rule == ContinuousSplitRule:
                self.X[:, idx] = jitter_duplicated(self.X[:, idx], np.nanstd(self.X[:, idx]))

        init_mean = self.Y.mean()
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))
        y_unique = np.unique(self.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.Y.std() / self.m**0.5

        self.running_sd = [RunningSd((self.leaves_shape, self.num_observations)) for _ in range(self.trees_shape)]

        self.sum_trees = np.full((self.trees_shape, self.leaves_shape, self.Y.shape[0]), init_mean).astype(config.floatX)
        self.sum_trees_noi = self.sum_trees - init_mean
        self.a_table = DecisionTable.new_table(
            leaf_node_value=np.full(self.leaves_shape, init_mean / self.m),
            idx_data_points=np.arange(self.num_observations, dtype="int32"),
            num_observations=self.num_observations,
            shape=self.leaves_shape,
            split_rules=self.split_rules,
        )

        self.normal = NormalSampler(1, self.leaves_shape)
        self.uniform = UniformSampler(0, 1)
        self.prior_prob_depth = compute_prior_probability(self.bart.alpha, self.bart.beta)  # Prior on depth (1 - prob_nonterminal)
        self.ssv = SampleSplittingVariable(self.alpha_vec)

        self.tune = True

        batch_0 = max(1, int(self.m * batch[0]))
        batch_1 = max(1, int(self.m * batch[1]))
        self.batch = (batch_0, batch_1)

        self.num_proposals = num_proposals

        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.likelihood_logp = logp(initial_point, [model.datalogp], [value_bart], shared)

        self.all_tables = [[self.a_table.copy() for _ in range(self.m)] for _ in range(self.trees_shape)]
        self.lower = 0
        self.iter = 0
        super().__init__([value_bart], shared)

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        accept_rate = 0.0

        upper = min(self.lower + self.batch[not self.tune], self.m)
        tree_ids = range(self.lower, upper)
        self.lower = upper if upper < self.m else 0

        for odim in range(self.trees_shape):
            for tree_id in tree_ids:
                self.iter += 1
                # Compute residuals without this table
                self.sum_trees_noi[odim] = self.sum_trees[odim] - self.all_tables[odim][tree_id]._predict()

                current_table = self.all_tables[odim][tree_id]
                current_logp = self.compute_logp(current_table, odim)

                accepts = 0
                for _ in range(self.num_proposals):
                    proposed_table = current_table.copy()
                    move_type = np.random.choice(['grow', 'prune', 'change'], p=[0.4, 0.4, 0.2])  # Balanced probabilities

                    if move_type == 'grow' and proposed_table.get_depth() < len(self.prior_prob_depth) - 1:
                        idx_split = self.ssv.rvs()
                        selected_predictor = self.available_predictors[idx_split]
                        avail_vals = self.X[:, selected_predictor]
                        split_rule = proposed_table.split_rules[selected_predictor]
                        split_value = split_rule.get_split_value(avail_vals)
                        if split_value is not None:
                            proposed_table.grow(selected_predictor, split_value, self.X, self.missing_data, self.sum_trees[odim], self.leaf_sd[odim], self.m, self.response, self.normal, self.leaves_shape)

                    elif move_type == 'prune' and proposed_table.get_depth() > 0:
                        proposed_table.prune()

                    elif move_type == 'change' and proposed_table.get_depth() > 0:
                        level_idx = np.random.randint(proposed_table.get_depth())
                        new_var = self.ssv.rvs()
                        selected_predictor = self.available_predictors[new_var]
                        avail_vals = self.X[:, selected_predictor]
                        split_rule = proposed_table.split_rules[selected_predictor]
                        split_value = split_rule.get_split_value(avail_vals)
                        if split_value is not None:
                            proposed_table.change_split(level_idx, selected_predictor, split_value)
                            proposed_table.grow(0, split_value, self.X, self.missing_data, self.sum_trees[odim], self.leaf_sd[odim], self.m, self.response, self.normal, self.leaves_shape)  # Recompute leaves after change

                    proposed_logp = self.compute_logp(proposed_table, odim)

                    log_ratio = proposed_logp - current_logp + self.transition_prob(current_table, proposed_table, move_type) - self.transition_prob(proposed_table, current_table, move_type[::-1])  # MH correction for proposal asymmetry

                    if log_ratio >= 0 or np.log(np.random.random()) < log_ratio:
                        current_table = proposed_table
                        current_logp = proposed_logp
                        accepts += 1

                accept_rate += accepts / self.num_proposals
                self.all_tables[odim][tree_id] = current_table
                new = current_table._predict()
                self.sum_trees[odim] = self.sum_trees_noi[odim] + new

                if self.tune:
                    if self.iter > self.m:
                        self.ssv = SampleSplittingVariable(self.alpha_vec)

                    for index in current_table.get_split_variables():
                        self.alpha_vec[index] += 1

                    if self.iter > 2:
                        self.leaf_sd[odim] = self.running_sd[odim].update(new)
                    else:
                        self.running_sd[odim].update(new)
                else:
                    for index in current_table.get_split_variables():
                        variable_inclusion[index] += 1

        if not self.tune:
            self.bart.all_trees.append(self.all_tables)  # Adapted for tables

        variable_inclusion = _encode_vi(variable_inclusion)

        stats = {"variable_inclusion": variable_inclusion, "tune": self.tune, "accept": accept_rate / (len(tree_ids) * self.trees_shape)}
        return self.sum_trees, [stats]

    def compute_logp(self, table, odim):
        """Compute log-posterior for table (likelihood + prior)."""
        delta = np.identity(self.trees_shape)[odim][:, None, None] * table._predict()[None, :, :]
        lik = self.likelihood_logp((self.sum_trees_noi + delta).flatten())
        depth = table.get_depth()
        prior = np.log(1 - self.prior_prob_depth[depth]) if depth < len(self.prior_prob_depth) - 1 else -10  # Penalize deep tables
        return lik + prior

    def transition_prob(self, from_table, to_table, move_type):
        """Log transition probability (simple uniform for now, can be improved)."""
        return 0.0  # Assume symmetric for simplicity; adjust for asymmetry if needed

    @staticmethod
    def competence(var, has_grad):
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
