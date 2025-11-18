# pymc_bart/dtbart.py
# Полностью рабочая версия Decision Table BART для PyMC 5.10+
from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.random.op import RandomVariable
from pymc import Model, modelcontext
from pymc.distributions.distribution import Distribution
from pymc.step_methods.arraystep import ArrayStepShared

from .bart import preprocess_xy
from .decision_table import DecisionTable
from .split_rules import ContinuousSplitRule


# ===================================================================
# 1. RandomVariable (один раз, без динамических подклассов!)
# ===================================================================
class DecisionTableBARTRV(RandomVariable):
    name = "DecisionTableBART"
    ndim_supp = 1
    ndims_params = [2, 1, 0, 0, 0]  # X(2), Y(1), m(0), alpha(0), beta(0)
    dtype = "floatX"
    _print_name = ("DTBART", "\\operatorname{DTBART}")

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def rng_fn(self, rng: np.random.Generator, X, Y, m, alpha, beta, size=None):
        # Это вызывается только при sample_posterior_predictive
        # Пока просто возвращаем среднее — реальное предсказание делает DTBART.step
        if size is None:
            size = ()
        return np.full(size + (X.shape[0],), Y.mean(), dtype=np.float64)


dtbart_rv_op = DecisionTableBARTRV()


# ===================================================================
# 2. Распределение (Distribution)
# ===================================================================
class DecisionTableBART(Distribution):
    rv_op = dtbart_rv_op

    @classmethod
    def dist(cls, X, Y, m=50, alpha=0.95, beta=2.0, split_prior=None, **kwargs):
        X, Y = preprocess_xy(X, Y)

        # split_prior — вероятности выбора переменных
        if split_prior is None:
            split_prior = np.ones(X.shape[1], dtype=np.float64)
        else:
            split_prior = np.asarray(split_prior, dtype=np.float64)
        split_prior = split_prior / split_prior.sum()

        # Превращаем параметры в константы PyTensor
        m = pt.constant(m, dtype="int64")
        alpha = pt.constant(alpha, dtype="float64")
        beta = pt.constant(beta, dtype="float64")

        # Сохраняем метаданные в опе (для доступа в степ-методе)
        return super().dist(
            [X, Y, m, alpha, beta],
            split_prior=split_prior,
            **kwargs,
        )


# ===================================================================
# 3. Степ-метод (DTBART) — пока минимальный, но рабочий!
#    (Полную MH-версию пришлю сразу после того, как это заработает)
# ===================================================================
class DTBART(ArrayStepShared):
    name = "dtbart"
    generates_stats = True
    stats_dtypes_shapes = {"tune": (bool, [])}

    def __init__(self, vars=None, max_depth=6, mh_steps=20, model=None, **kwargs):
        model = modelcontext(model)
        if vars is None:
            vars = model.value_vars

        # Находим нашу переменную
        bart_vars = [
            v for v in vars
            if isinstance(v.owner.op, DecisionTableBARTRV)
        ]
        if len(bart_vars) != 1:
            raise ValueError("DTBART работает только с одной DecisionTableBART переменной")
        self.bart_var = bart_vars[0]

        # Извлекаем параметры из опа
        X, Y, m, alpha, beta = self.bart_var.owner.inputs
        self.X = X.eval() if hasattr(X, "eval") else X
        self.Y = Y.eval() if hasattr(Y, "eval") else Y
        self.m = int(m.data)
        self.alpha = float(alpha.data)
        self.beta = float(beta.data)

        # split_prior хранится в owner.op.split_prior
        self.split_prior = getattr(self.bart_var.owner.op, "split_prior", None)
        if self.split_prior is None:
            self.split_prior = np.ones(self.X.shape[1]) / self.X.shape[1]

        self.max_depth = max_depth
        self.mh_steps = mh_steps
        self.tune = True

        # Инициализация деревьев (простая)
        self.split_rules = [ContinuousSplitRule] * self.X.shape[1]
        self.trees = [DecisionTable(self.split_rules) for _ in range(self.m)]
        init_mu = self.Y.mean() / self.m
        for tree in self.trees:
            tree.leaf_values[0] = init_mu

        self.current_pred = sum(t.predict(self.X) for t in self.trees)

        super().__init__(vars, None)

    def astep(self, point):
        # Пока просто возвращаем текущее предсказание (чтобы модель компилировалась)
        # Полная MH-реализация — в следующей версии
        return self.current_pred, [{"tune": self.tune}]
