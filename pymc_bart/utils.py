# Добавьте эту функцию в существующий utils.py

def _sample_posterior(
    all_trees: list[list],
    X: TensorLike,
    rng: np.random.Generator,
    size: int | tuple[int, ...] | None = None,
    excluded: list[int] | None = None,
    shape: int = 1,
) -> npt.NDArray:
    """
    Generate samples from the BART-posterior.

    Parameters
    ----------
    all_trees : list
        List of all trees sampled from a posterior
    X : tensor-like
        A covariate matrix. Use the same used to fit BART for in-sample predictions or a new one for
        out-of-sample predictions.
    rng : NumPy RandomGenerator
    size : int or tuple
        Number of samples.
    excluded : Optional[npt.NDArray[np.int_]]
        Indexes of the variables to exclude when computing predictions
    """
    stacked_trees = all_trees

    if isinstance(X, Variable):
        X = X.eval()

    if size is None:
        size_iter: list | tuple = (1,)
    elif isinstance(size, int):
        size_iter = [size]
    else:
        size_iter = size

    flatten_size = 1
    for s in size_iter:
        flatten_size *= s

    idx = rng.integers(0, len(stacked_trees), size=flatten_size)

    # Check if we're using decision tables
    if (stacked_trees and stacked_trees[0] and 
        hasattr(stacked_trees[0][0], '__class__') and 
        stacked_trees[0][0].__class__.__name__ == 'DecisionTable'):
        # Handle decision tables
        trees_shape = len(stacked_trees[0])
        leaves_shape = shape // trees_shape

        pred = np.zeros((flatten_size, trees_shape, leaves_shape, X.shape[0]))

        for ind, p in enumerate(pred):
            for odim, odim_trees in enumerate(stacked_trees[idx[ind]]):
                for table in odim_trees:
                    p[odim] += table.predict(x=X, excluded=excluded, shape=leaves_shape)

        return pred.transpose((0, 3, 1, 2)).reshape((*size_iter, -1, shape))
    else:
        # Original tree handling
        trees_shape = len(stacked_trees[0])
        leaves_shape = shape // trees_shape

        pred = np.zeros((flatten_size, trees_shape, leaves_shape, X.shape[0]))

        for ind, p in enumerate(pred):
            for odim, odim_trees in enumerate(stacked_trees[idx[ind]]):
                for tree in odim_trees:
                    p[odim] += tree.predict(x=X, excluded=excluded, shape=leaves_shape)

        return pred.transpose((0, 3, 1, 2)).reshape((*size_iter, -1, shape))
