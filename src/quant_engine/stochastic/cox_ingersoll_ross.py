import numpy as np
from .brownian_motion import brownian_motion

def cox_ingersoll_ross(
    T=1.0,
    N=1000,
    M=1,
    theta=0.5,
    mu=0.02,
    sigma=0.1,
    X0=0.01,
    seed=None
):
    """
    Simulate Cox-Ingersoll-Ross (CIR) processes.

    The CIR process satisfies the SDE:
        dX_t = theta * (mu - X_t) dt + sigma * sqrt(X_t) dW_t

    Parameters
    ----------
    T : float, default=1.0
        Time horizon
    N : int, default=1000
        Number of time steps
    M : int, default=1
        Number of simulated paths
    theta : float, default=0.5
        Mean-reversion rate
    mu : float, default=0.02
        Long-term mean
    sigma : float, default=0.1
        Volatility coefficient
    X0 : float, default=0.01
        Initial value
    seed : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    t : np.ndarray
        Time grid of length N+1
    X : np.ndarray
        Simulated paths of shape (M, N+1)
    """

    t, W = brownian_motion(T, N, M, seed=seed)
    dt = T / N

    # Initialize paths
    X = np.empty_like(W)
    X[:, 0] = X0 if M > 1 else X0

    # Full truncation Euler scheme to ensure positivity
    for i in range(1, N+1):
        sqrt_X_prev = np.sqrt(np.maximum(X[:, i-1], 0))
        X[:, i] = X[:, i-1] + theta * (mu - X[:, i-1]) * dt + sigma * sqrt_X_prev * (W[:, i] - W[:, i-1])
        X[:, i] = np.maximum(X[:, i], 0)  # enforce non-negativity

    return t, X
