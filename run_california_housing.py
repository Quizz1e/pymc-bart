"""
Script to run PyMC BART with Decision Tables (MH sampler) on California Housing dataset.
Computes RMSE, R², and coverage metrics on test set.
"""

import numpy as np
import pymc as pm
import pymc_bart as pmb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def sample_posterior_decision_tables(
    all_tables: list[list[pmb.DecisionTable]],
    X: np.ndarray,
    rng: np.random.Generator,
    size: int = 100,
    excluded: list[int] | None = None,
) -> np.ndarray:
    """
    Generate predictions from DecisionTable posterior (similar to _sample_posterior for Tree).
    
    Parameters
    ----------
    all_tables : list[list[DecisionTable]]
        List of all decision tables sampled from posterior (each inner list is one iteration)
    X : np.ndarray
        Input data for prediction
    rng : np.random.Generator
        Random number generator
    size : int
        Number of posterior samples to draw
    excluded : Optional[list[int]]
        Variables to exclude from prediction
        
    Returns
    -------
    np.ndarray
        Predictions of shape (size, n_observations)
    """
    if not all_tables:
        raise ValueError("all_tables is empty")
    
    if size is None or size <= 0:
        size = 1
    
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_obs = X.shape[0]
    predictions = np.zeros((size, n_obs))
    
    # Randomly select iterations
    idx = rng.integers(0, len(all_tables), size=size)
    
    for i, iter_idx in enumerate(idx):
        # Get tables from this iteration
        tables = all_tables[iter_idx]
        
        if not tables:
            raise ValueError(f"No tables found at iteration {iter_idx}")
        
        # Sum predictions from all tables (ensemble)
        pred_sum = np.zeros(n_obs)
        for table in tables:
            try:
                pred = table.predict(X, excluded=excluded, shape=1)
                # Handle different output shapes
                if pred.ndim > 1:
                    pred = pred.ravel()
                # Ensure pred has the right length
                if len(pred) != n_obs:
                    raise ValueError(
                        f"Prediction length {len(pred)} doesn't match n_obs {n_obs}"
                    )
                pred_sum += pred
            except Exception as e:
                raise RuntimeError(f"Error predicting from table: {e}") from e
        
        predictions[i] = pred_sum
    
    return predictions


def compute_coverage(y_true: np.ndarray, y_pred_samples: np.ndarray, ci_level: float = 0.95) -> float:
    """
    Compute coverage: fraction of points where true value falls within credible interval.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred_samples : np.ndarray
        Posterior prediction samples of shape (n_samples, n_observations)
    ci_level : float
        Credible interval level (default 0.95)
        
    Returns
    -------
    float
        Coverage fraction
    """
    lower = np.percentile(y_pred_samples, (1 - ci_level) / 2 * 100, axis=0)
    upper = np.percentile(y_pred_samples, (1 + ci_level) / 2 * 100, axis=0)
    
    in_interval = (y_true >= lower) & (y_true <= upper)
    return in_interval.mean()


def main():
    """Main function to run the model."""
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_seed=42
    )
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print("\nBuilding PyMC model with BART (Decision Tables)...")
    
    # Create PyMC model
    with pm.Model() as model:
        # BART component
        mu = pmb.BART(
            "mu",
            X_train_scaled,
            y_train_scaled,
            m=50,  # Number of decision tables
        )
        
        # Error term
        sigma = pm.HalfNormal("sigma", 1.0)
        
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_scaled)
        
        # Create MH sampler for Decision Tables
        # Use smaller leaf_sd for better regularization
        # Estimate initial sigma from data
        initial_sigma = np.std(y_train_scaled)
        step = pmb.MHDecisionTableSampler(
            vars=[mu],
            num_tables=50,
            move_probs=(0.25, 0.25, 0.5),  # (grow, prune, change) - favor change moves
            move_adapt_rate=0.1,
            move_prob_prior=0.05,
            leaf_sd=0.1 * initial_sigma,  # Smaller leaf_sd for regularization
            n_jobs=1,
            rng_seed=42,
        )
        
        print("Sampling from posterior...")
        print(f"Initial sigma estimate: {initial_sigma:.4f}")
        print(f"Leaf SD: {step.leaf_sd:.4f}")
        # Sample from posterior
        idata = pm.sample(
            tune=200,  # Warm-up iterations (reduced for faster testing)
            draws=200,  # Posterior samples (reduced for faster testing)
            step=step,
            chains=1,
            random_seed=42,
            return_inferencedata=True,
        )
    
    print("\nSampling completed!")
    print(f"Posterior samples shape: {idata.posterior['mu'].shape}")
    
    # Get DecisionTable objects from sampler
    # The tables are stored in the sampler's all_tables attribute
    sampler = step
    all_tables = sampler.all_tables
    
    print(f"\nNumber of iterations with tables: {len(all_tables)}")
    print(f"Number of tables per iteration: {len(all_tables[0]) if all_tables else 0}")
    
    # Note: all_tables includes initial state (index 0) + posterior draws only
    # (tune iterations are not stored because we only save when self.tune == False)
    # So we can use all entries except the initial state
    if len(all_tables) > 1:
        # Skip the initial state (index 0), use all posterior draws
        posterior_tables = all_tables[1:]
    else:
        # Fallback: use all tables if somehow only initial state exists
        posterior_tables = all_tables
    
    if not posterior_tables:
        raise ValueError("No posterior tables found for predictions")
    
    print(f"Using {len(posterior_tables)} posterior iterations for predictions")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    rng = np.random.default_rng(42)
    
    # Sample from posterior for predictions
    n_pred_samples = 200
    y_pred_samples = sample_posterior_decision_tables(
        all_tables=posterior_tables,
        X=X_test_scaled,
        rng=rng,
        size=n_pred_samples,
        excluded=None,
    )
    
    # Compute mean predictions
    y_pred_mean = y_pred_samples.mean(axis=0)
    
    # Transform back to original scale
    y_pred_mean_original = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_mean_original))
    r2 = r2_score(y_test_original, y_pred_mean_original)
    
    # Transform prediction samples back to original scale for coverage
    y_pred_samples_original = scaler_y.inverse_transform(
        y_pred_samples.reshape(-1, 1)
    ).reshape(y_pred_samples.shape)
    
    # Compute coverage (on original scale)
    coverage = compute_coverage(y_test_original, y_pred_samples_original, ci_level=0.95)
    
    print("\n" + "="*60)
    print("METRICS ON TEST SET")
    print("="*60)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Coverage (95% CI): {coverage:.4f} ({coverage*100:.2f}%)")
    print("="*60)
    
    # Print some diagnostics
    print("\nSampler diagnostics:")
    print(f"  Accept rate: {idata.sample_stats['accept_rate'].mean().item():.4f}")
    print(f"  Move types: {idata.sample_stats['move_type'].values}")
    
    return {
        "rmse": rmse,
        "r2": r2,
        "coverage": coverage,
        "idata": idata,
        "y_pred_mean": y_pred_mean_original,
        "y_test": y_test_original,
    }


if __name__ == "__main__":
    results = main()

