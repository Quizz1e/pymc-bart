#   Copyright 2022 The PyMC Developers
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

from abc import abstractmethod
from typing import Mapping

import numpy as np
from numba import njit


class SplitRule:
    """
    Abstract template class for a split rule
    """

    @staticmethod
    @abstractmethod
    def get_split_value(available_splitting_values):
        pass

    @staticmethod
    @abstractmethod
    def divide(available_splitting_values, split_value):
        pass


class ContinuousSplitRule(SplitRule):
    """
    Standard continuous split rule: pick a pivot value and split
    depending on if variable is smaller or greater than the value picked.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1:
            idx_selected_splitting_values = int(
                np.random.random() * len(available_splitting_values)
            )
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        return available_splitting_values <= split_value


class OneHotSplitRule(SplitRule):
    """Choose a single categorical value and branch on if the variable is that value or not"""

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(
            available_splitting_values == available_splitting_values[0]
        ):
            idx_selected_splitting_values = int(
                np.random.random() * len(available_splitting_values)
            )
            split_value = available_splitting_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        return available_splitting_values == split_value


class SubsetSplitRule(SplitRule):
    """
    Choose a random subset of the categorical values and branch on belonging to that set.
    This is the approach taken by Sameer K. Deshpande.
    flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
    `link <https://arxiv.org/abs/2211.04459>`__
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1 and not np.all(
            available_splitting_values == available_splitting_values[0]
        ):
            unique_values = np.unique(available_splitting_values)
            while True:
                sample = np.random.randint(0, 2, size=len(unique_values)).astype(bool)
                if np.any(sample):
                    break
            split_value = unique_values[sample]
        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        return np.isin(available_splitting_values, split_value)


class TargetMeanSplitRule(SplitRule):
    """
    CatBoost-style target statistics split rule for categorical features.

    This rule relies on precomputed target summaries per categorical value.
    Users must call ``set_target_statistics`` before sampling so the rule can
    translate raw categorical values into smoothed target means.
    """

    _value_sum: dict = {}
    _value_count: dict = {}
    _prior_mean: float = 0.0
    _prior_weight: float = 1.0

    @classmethod
    def set_target_statistics(
        cls,
        summaries: Mapping,
        prior_mean: float | None = None,
        prior_weight: float = 1.0,
    ) -> None:
        """
        Configure category target statistics.

        Parameters
        ----------
        summaries : Mapping
            Dictionary mapping categorical values to tuples ``(target_sum, count)``
            or to precomputed means. Values can be anything hashable (ints/strings).
        prior_mean : Optional[float]
            Global mean used for smoothing unseen categories. If ``None``, defaults
            to the weighted average of provided summaries. Defaults to ``None``.
        prior_weight : float
            Strength of the prior mean when smoothing per-category statistics.
            Larger values shrink category means closer to ``prior_mean``.
        """
        if not summaries:
            raise ValueError("summaries must contain at least one category.")

        sums = {}
        counts = {}
        derived_sum = 0.0
        derived_count = 0.0

        for key, value in summaries.items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                target_sum, count = value
            else:
                # Assume value is already a mean; treat count as 1.
                target_sum = float(value)
                count = 1.0
            target_sum = float(target_sum)
            count = float(count)
            if count <= 0:
                continue
            sums[key] = target_sum
            counts[key] = count
            derived_sum += target_sum
            derived_count += count

        if not sums:
            raise ValueError("All summaries had non-positive counts.")

        cls._value_sum = sums
        cls._value_count = counts
        cls._prior_weight = max(0.0, float(prior_weight))

        if prior_mean is not None:
            cls._prior_mean = float(prior_mean)
        elif derived_count > 0:
            cls._prior_mean = derived_sum / derived_count
        else:
            cls._prior_mean = 0.0

    @classmethod
    def _encode_with_target_mean(cls, values: np.ndarray) -> np.ndarray:
        if not cls._value_sum:
            raise RuntimeError(
                "TargetMeanSplitRule: call set_target_statistics before using this rule."
            )

        encoded = np.empty(values.shape[0], dtype=np.float64)
        for idx, val in enumerate(values):
            target_sum = cls._value_sum.get(val)
            count = cls._value_count.get(val)
            if target_sum is None or count is None or count <= 0:
                encoded[idx] = cls._prior_mean
            else:
                encoded[idx] = (target_sum + cls._prior_mean * cls._prior_weight) / (
                    count + cls._prior_weight
                )
        return encoded

    @staticmethod
    def get_split_value(available_splitting_values):
        encoded = TargetMeanSplitRule._encode_with_target_mean(available_splitting_values)
        if encoded.size <= 1 or np.all(encoded == encoded[0]):
            return None
        idx_selected = int(np.random.random() * encoded.size)
        return encoded[idx_selected]

    @staticmethod
    def divide(available_splitting_values, split_value):
        encoded = TargetMeanSplitRule._encode_with_target_mean(available_splitting_values)
        return encoded <= split_value