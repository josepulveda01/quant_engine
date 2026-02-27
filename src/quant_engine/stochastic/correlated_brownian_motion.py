import numpy as np
from .brownian_motion import brownian_motion

def correlated_brownian_motion(
    T: float,
    N: int,
    M: int,
    corr_matrix: np.ndarray,
    seed: int | None = None
):
    """
    Simulate correlated Brownian motions.

    Parameters
    ----------
    T : float
        Time horizon
    N : int
        Number of steps
    M : int
        Number of paths
    corr_matrix : np.ndarray, shape (d, d)
        Correlation matrix of the d Brownian motions
    seed : int | None
        Random seed

    Returns
    -------
    t : np.ndarray
        Time grid of length N+1
    W : np.ndarray
        Simulated paths of shape (M, N+1, d)
    """

    d = corr_matrix.shape[0]  # number of Brownian components

    # Check that correlation matrix is valid
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("corr_matrix must be square")
    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("corr_matrix must be symmetric")
    if not np.all(np.linalg.eigvals(corr_matrix) >= 0):
        raise ValueError("corr_matrix must be positive semi-definite")

    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent Brownian motions
    t, W_indep = brownian_motion(T, N, M*d, seed=seed)

    # Reshape to (M, N+1, d)
    W_indep = W_indep.reshape(M, N+1, d)

    # Apply correlation
    W_corr = np.matmul(W_indep, L.T)  # shape (M, N+1, d)

    return t, W_corr
