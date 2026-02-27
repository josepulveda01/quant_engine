import numpy as np
from .brownian_motion import brownian_motion

def ornstein_uhlenbeck(
    T=1.0,
    N=1000,
    M=1,
    theta=0.7,
    mu=1.0,
    sigma=0.3,
    X0=0.0,
    seed=None
):
    """
    Simulate Ornstein-Uhlenbeck (O-U) processes.

    The O-U process satisfies the SDE:
        dX_t = theta * (mu - X_t) dt + sigma dW_t

    Parameters
    ----------
    T : float, default=1.0
        Time horizon
    N : int, default=1000
        Number of time steps
    M : int, default=1
        Number of simulated paths
    theta : float, default=0.7
        Mean-reversion rate
    mu : float, default=1.0
        Long-term mean
    sigma : float, default=0.3
        Volatility coefficient
    X0 : float, default=0.0
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

    # Euler-Maruyama scheme
    for i in range(1, N+1):
        X[:, i] = X[:, i-1] + theta * (mu - X[:, i-1]) * dt + sigma * (W[:, i] - W[:, i-1])

    return t, X
