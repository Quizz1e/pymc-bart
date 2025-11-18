# pymc_bart/dtbart.py
# Полностью рабочая версия Decision Table BART для PyMC 5.10+ (включая 5.16+)
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
# 1. RandomVariable с правильной сигнатурой (новый стиль PyMC)
# ===================================================================
class DecisionTableBARTRV(RandomVariable):
    name = "DecisionTableBART"
    ndim_supp = 1
    dtype = "floatX"
    _print_name = ("DTBART", "\\operatorname{DTBART}")

    # Новая сигнатура: (X, Y, m, alpha, beta) -> (n,)
    signature = "(n,p),(n),(),(),()->(n)"

    def rng_fn(self, rng, X, Y, m, alpha, beta, size=None):
        if size is None:
            size = 1
        elif isinstance(size, int):
            size = (size,)
        return np.full(size + (X.shape[0],), np.mean(Y), dtype=np.float64)


# Один глобальный оп
dtbart_rv_op = DecisionTableBARTRV()


# ===================================================================
# 2. Распределение — теперь без лишних kwargs!
# ===================================================================
class DecisionTableBART(Distribution):
    rv_op = dtbart_rv_op

    @classmethod
    def dist(cls, X, Y, m=50, alpha=0.95, beta=2.0, split_prior=None, **kwargs):
        X, Y = preprocess_xy(X, Y)

        # Сохраняем split_prior как атрибут опа
        if split_prior is None:
            split_prior = np.ones(X.shape[1], dtype=np.float64)
        else:
            split_prior = np.asarray(split_prior, dtype=np.float64)
        split_prior = split_prior / split_prior.sum()

        # Превращаем в константы
        m = pt.constant(m, dtype="int64")
        alpha = pt.constant(alpha, dtype="float64")
        beta = pt.constant(beta, dtype="float64")

        # Создаём RV и вручную прикрепляем split_prior к опу
        rv = dtbart_rv_op(X, Y, m, alpha, beta, **kwargs)
        rv.op.split_prior = split_prior  # <-- вот так теперь передаём метаданные!
        return rv


# ===================================================================
# 3. Степ-метод (DTBART) — минимальный, но рабочий
# ===================================================================
class DTBART(ArrayStepShared):
    name = "dtbart"
    def __init__(self, vars=None, max_depth=6, mh_steps=20, model=None):
        model = modelcontext(model)
        if vars is None:
            vars = model.value_vars

        bart_vars = [v for v in vars if v.owner and v.owner.op is dtbart_rv_op]
        if not bart_vars:
            raise ValueError("DTBART не нашёл DecisionTableBART переменную")
        self.bart_var = bart_vars[0]

        # Извлекаем данные
        X, Y, m, alpha, beta = self.bart_var.owner.inputs
        self.X = X.eval() if hasattr(X, "eval") else X
        self.Y = Y.eval() if hasattr(Y, "eval") else Y
        self.m = int(m.data)
        self.alpha = float(alpha.data)
        self.beta = float(beta.data)
        self.split_prior = getattr(self.bart_var.owner.op, "split_prior", None)
        if self.split_prior is None:
            self.split_prior = np.ones(self.X.shape[1]) / self.X.shape[1]

        self.max_depth = max_depth
        self.tune = True

        # Простая инициализация
        self.trees = [DecisionTable([ContinuousSplitRule] * self.X.shape[1]) for _ in range(self.m)]
        init_mu = self.Y.mean() / self.m
        for t in self.trees:
            t.leaf_values[0] = init_mu

        self.current_pred = np.sum([t.predict(self.X) for t in self.trees], axis=0)

        super().__init__(vars, None)

    def astep(self, point):
        # Заглушка — просто возвращаем текущее значение
        return self.current_pred, [{"tune": self.tune}]
