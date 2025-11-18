# pymc_bart/dtbart.py
from __future__ import annotations

import warnings
from multiprocessing import Manager

import numpy as np
import pytensor.tensor as pt
from numba import njit
from pymc import Model, modelcontext
from pymc.distributions.distribution import Distribution
from pymc.initial_point import PointType
from pymc.logprob.abstract import _logprob
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pytensor.tensor.random.op import RandomVariable

from .bart import preprocess_xy
from .decision_table import DecisionTable
from .split_rules import ContinuousSplitRule
from .utils import _sample_posterior


class DecisionTableBARTRV(RandomVariable):
    name = "DecisionTableBART"
    signature = "(m,n),(m),(),(),() -> (m)"
    dtype = "floatX"
    _print_name = ("DTBART", "\\operatorname{DTBART}")

    @classmethod
    def rng_fn(cls, rng, X, Y, m, alpha, beta, size=None):
        # этот метод вызывается только для инициализации и предсказаний
        if not hasattr(cls, "all_trees") or not cls.all_trees:
            return np.full((size or 1, Y.shape[0]), Y.mean())
        return _sample_posterior(cls.all_trees, X, rng=rng, shape=size).squeeze().T


dtbart_rv = DecisionTableBARTRV()


class DecisionTableBART(Distribution):
    rv_op = dtbart_rv

    @classmethod
    def dist(cls, X, Y, m=50, alpha=0.95, beta=2.0, split_prior=None, **kwargs):
        X, Y = preprocess_xy(X, Y)
        split_prior = np.array([]) if split_prior is None else np.asarray(split_prior)

        manager = Manager()
        all_trees = manager.list()

        op = type(
            "DecisionTableBARTInstance",
            (DecisionTableBARTRV,),
            {
                "all_trees": all_trees,
                "X": X,
                "Y": Y,
                "m": m,
                "alpha": alpha,
                "beta": beta,
                "split_prior": split_prior,
            },
        )()
        return super().dist([X, Y, m, alpha, beta], op=op, **kwargs)


class DTBART(ArrayStepShared):
    name = "dtbart"
    generates_stats = True
    stats_dtypes_shapes = {"variable_inclusion": (object, []), "tune": (bool, [])}

    def __init__(
        self,
        vars=None,
        max_depth: int = 8,
        mh_steps: int = 20,
        alpha: float = 0.95,
        beta: float = 2.0,
        split_prior=None,
        model=None,
        **kwargs,
    ):
        model = modelcontext(model)
        if vars is None:
            vars = model.value_vars

        # находим BART-переменную (теперь это DecisionTableBART)
        bart_vars = [
            v for v in vars
            if model.values_to_rvs[v].owner.op.__class__.__name__ == "DecisionTableBARTInstance"
        ]
        if len(bart_vars) != 1:
            raise ValueError("DTBART работает только с одной DecisionTableBART переменной")

        value_var = bart_vars[0]
        self.bart_op = model.values_to_rvs[value_var].owner.op

        self.X = self.bart_op.X if not isinstance(self.bart_op.X, pt.TensorVariable) else self.bart_op.X.eval()
        self.Y = self.bart_op.Y if not isinstance(self.bart_op.Y, pt.TensorVariable) else self.bart_op.Y.eval()
        self.m = self.bart_op.m
        self.alpha = self.bart_op.alpha
        self.beta = self.bart_op.beta

        self.split_prior = (np.ones(self.X.shape[1]) if split_prior is None else np.asarray(split_prior))
        self.split_prior = self.split_prior / self.split_prior.sum()

        self.split_rules = [ContinuousSplitRule] * self.X.shape[1]   # можно расширить позже

        self.max_depth = max_depth
        self.mh_steps = mh_steps

        # кэш квантилей для выбора порогов
        self.threshold_candidates = [
            np.quantile(self.X[:, j], np.linspace(0.05, 0.95, 50)) for j in range(self.X.shape[1])
        ]

        # prior p(split | depth)
        self.prior_split = np.array(
            [alpha * (1 + d) ** (-beta) for d in range(max_depth + 2)]
        )
        self.prior_leaf = 1 - self.prior_split

        # инициализация деревьев
        init_mu = self.Y.mean() / self.m
        self.trees: list[DecisionTable] = [
            DecisionTable(self.split_rules) for _ in range(self.m)
        ]
        for tree in self.trees:
            tree.leaf_values[0] = init_mu

        self.sum_trees_pred = sum(tree.predict(self.X) for tree in self.trees)

        # для leaf prior variance (как в оригинальном BART)
        self.leaf_sigma = self.Y.std() / np.sqrt(self.m)

        # likelihood logp функция
        shared = make_shared_replacements({value_var: value_var}, [value_var], model)
        self.logp_func = logp({value_var.name: value_var}, [model.datalogp], [value_var], shared)

        # для тюнинга и статистики
        self.tune = True
        self.variable_inclusion = np.zeros(self.X.shape[1])

        super().__init__(vars, shared)

    def astep(self, _):
        # один шаг — обновляем все деревья по очереди
        for tree in self.trees:
            current_pred = tree.predict(self.X)
            residual = self.Y - (self.sum_trees_pred - current_pred)

            for _ in range(self.mh_steps):
                proposal = tree.copy()

                # выбор мува
                if proposal.depth >= self.max_depth:
                    grow_p, prune_p, change_p = 0.0, 0.5, 0.5
                elif proposal.depth == 0:
                    grow_p, prune_p, change_p = 0.5, 0.0, 0.5
                else:
                    grow_p, prune_p, change_p = 0.25, 0.25, 0.5

                move = np.random.choice(["grow", "prune", "change"], p=[grow_p, prune_p, change_p])

                log_q_ratio = 0.0
                log_prior_ratio = 0.0

                if move == "grow" and proposal.depth < self.max_depth:
                    feature = np.random.choice(len(self.split_prior), p=self.split_prior)
                    thr = np.random.choice(self.threshold_candidates[feature])
                    proposal.grow(feature, thr)

                    log_prior_ratio = np.log(self.prior_split[tree.depth])
                    log_q_ratio = np.log(grow_p) - np.log(prune_p)  # reverse = prune

                elif move == "prune" and proposal.depth > 0:
                    proposal.prune()
                    old_feature = tree.features[-1]
                    log_prior_ratio = np.log(self.prior_leaf[tree.depth])
                    log_q_ratio = np.log(prune_p) - np.log(grow_p)

                else:  # change
                    level = np.random.randint(proposal.depth)
                    old_feature = proposal.features[level]
                    new_feature = np.random.choice(len(self.split_prior), p=self.split_prior)
                    new_thr = np.random.choice(self.threshold_candidates[new_feature])
                    proposal.change_level(level, new_feature, new_thr)
                    if old_feature != new_feature:
                        log_prior_ratio = np.log(self.split_prior[new_feature] / self.split_prior[old_feature])

                # новые значения листьев (по частичным остаткам + шум)
                leaf_idx = proposal.get_leaf_indices(self.X)
                n_leaves = proposal.num_leaves()
                new_leaf_vals = np.zeros((n_leaves, self.Y.shape[1] if self.Y.ndim > 1 else 1))

                for leaf in range(n_leaves):
                    mask = leaf_idx == leaf
                    if mask.sum() > 0:
                        mu = residual[mask].mean()
                        new_leaf_vals[leaf] = mu / self.m + np.random.normal(0, self.leaf_sigma, size=new_leaf_vals.shape[1] or 1)
                    else:
                        new_leaf_vals[leaf] = proposal.leaf_values[leaf]  # не меняем пустые

                old_leaf_vals = proposal.leaf_values.copy()
                proposal.leaf_values = new_leaf_vals

                # log-likelihood ratio
                new_total_pred = self.sum_trees_pred - current_pred + proposal.predict(self.X)
                loglik_new = self.logp_func(new_total_pred.ravel())
                loglik_old = self.logp_func(self.sum_trees_pred.ravel())

                log_accept = loglik_new - loglik_old + log_prior_ratio + log_q_ratio

                if np.log(np.random.random()) < log_accept:
                    tree = proposal
                    current_pred = proposal.predict(self.X)
                    self.sum_trees_pred = new_total_pred

            # статистика variable inclusion (только после тюнинга)
            if not self.tune:
                for f in set(tree.features):
                    self.variable_inclusion[f] += 1

        # сохраняем деревья для posterior sampling
        self.bart_op.all_trees.append([t.copy() for t in self.trees])

        stats = {
            "variable_inclusion": _encode_vi(self.variable_inclusion.tolist()),
            "tune": self.tune,
        }
        return self.sum_trees_pred, [stats]
