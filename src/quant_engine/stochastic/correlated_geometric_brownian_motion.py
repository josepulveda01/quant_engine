import numpy as np
from .correlated_brownian_motion import correlated_brownian_motion

def correlated_geometric_brownian_motion(
    T: float,
    N: int,
    M: int,
    corr_matrix,
    S0: float = 1.0,
    mu: np.ndarray | float = 0.05,
    sigma: np.ndarray | float = 0.2,
    seed: int | None = None
):
    """
    Simulate correlated geometric Brownian motion (CGBM) paths.

    Parameters
    ----------
    T : float, default=1.0
        Time horizon.
    N : int, default=1000
        Number of time steps.
    M : int, default=1
        Number of simulated paths.
    S0 : float, default=1.0
        Initial asset price (or vector of initial prices if multiple assets).
    mu : float or np.ndarray, default=0.05
        Drift coefficient(s), scalar or length d.
    sigma : float or np.ndarray, default=0.2
        Volatility coefficient(s), scalar or length d.
    corr_matrix : np.ndarray, shape (d, d), default=[[1.0]]
        Correlation matrix between d assets.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time grid of length N+1
    S : np.ndarray
        Simulated correlated GBM paths of shape (M, N+1, d)
    """

    d = corr_matrix.shape[0]

    # Ensure mu and sigma are arrays of length d
    mu = np.full(d, mu) if np.isscalar(mu) else np.asarray(mu)
    sigma = np.full(d, sigma) if np.isscalar(sigma) else np.asarray(sigma)

    # Generate correlated Brownian motions
    t, W_corr = correlated_brownian_motion(T, N, M, corr_matrix, seed=seed)

    # Apply GBM formula
    drift = (mu - 0.5 * sigma**2) * t  # shape (N+1,) broadcastable
    S = S0 * np.exp(drift + W_corr * sigma)  # broadcasting to (M, N+1, d)

    return t, S
