#!/usr/bin/env python3
"""Train the MH-based Decision Table sampler on a real dataset and report metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.utils import _sample_posterior
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PyMC-BART MH sampler on the California Housing dataset."
    )
    parser.add_argument("--num-tables", type=int, default=50, help="Number of trees in the ensemble.")
    parser.add_argument("--leaf-sd", type=float, default=1.0, help="Leaf-value proposal scale.")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps.")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel threads for per-tree updates.")
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=500,
        help="Number of posterior predictive samples for evaluation.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.95,
        help="Central credible interval level used for coverage.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for data splitting and posterior sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save metrics as a JSON file.",
    )
    return parser.parse_args()


def load_dataset(test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = fetch_california_housing()
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pm.backends.base.MultiTrace | None, pm.model.FreeRV]:
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        bart = pmb.BART("bart", X_train, y_train, m=args.num_tables)

        pm.Normal("y_obs", mu=bart, sigma=sigma, observed=y_train)

        step = pmb.MHDecisionTableSampler(
            num_tables=args.num_tables,
            leaf_sd=args.leaf_sd,
            n_jobs=args.n_jobs,
            model=model,
        )

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.random_state,
            step=step,
            target_accept=0.9,
            progressbar=True,
            return_inferencedata=True,
        )

    return idata, bart


def sample_mu_predictions(
    bart_rv: pm.model.FreeRV,
    X_new: np.ndarray,
    num_samples: int,
    random_seed: int,
) -> np.ndarray:
    all_trees = list(bart_rv.owner.op.all_trees)
    if not all_trees:
        raise RuntimeError(
            "No stored trees were found on the BART operator. "
            "Ensure the MHDecisionTableSampler ran before calling this script."
        )

    rng = np.random.default_rng(random_seed)
    preds = _sample_posterior(
        all_trees=all_trees,
        X=X_new,
        rng=rng,
        size=num_samples,
        shape=1,
    )

    return np.asarray(preds).reshape(num_samples, X_new.shape[0])


def build_posterior_predictive(
    mu_samples: np.ndarray,
    sigma_draws: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    flattened_sigma = sigma_draws.reshape(-1)
    idx = rng.integers(0, flattened_sigma.size, size=mu_samples.shape[0])
    sigma_subset = flattened_sigma[idx][:, None]
    noise = rng.normal(0.0, sigma_subset, size=mu_samples.shape)
    return mu_samples + noise


def compute_metrics(
    y_true: np.ndarray,
    predictive_samples: np.ndarray,
    interval: float,
) -> dict[str, float]:
    pred_mean = predictive_samples.mean(axis=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred_mean)))
    r2 = float(r2_score(y_true, pred_mean))

    alpha = 1.0 - interval
    lower = np.quantile(predictive_samples, alpha / 2, axis=0)
    upper = np.quantile(predictive_samples, 1.0 - alpha / 2, axis=0)
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

    return {"rmse": rmse, "r2": r2, "coverage": coverage}


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = load_dataset(args.test_size, args.random_state)

    idata, bart_rv = fit_model(X_train, y_train, args)
    rng = np.random.default_rng(args.random_state)

    mu_samples = sample_mu_predictions(
        bart_rv=bart_rv,
        X_new=X_test,
        num_samples=args.posterior_samples,
        random_seed=args.random_state,
    )

    sigma_draws = idata.posterior["sigma"].values
    predictive_samples = build_posterior_predictive(mu_samples, sigma_draws, rng)

    metrics = compute_metrics(y_test, predictive_samples, args.interval)

    print("Evaluation metrics on California Housing test split:")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  R^2      : {metrics['r2']:.4f}")
    print(f"  Coverage : {metrics['coverage'] * 100:.2f}% (central {args.interval:.0%} interval)")

    if args.output is not None:
        import json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fout:
            json.dump(metrics, fout, indent=2)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()

