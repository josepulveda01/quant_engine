import numpy as np
from .brownian_motion import brownian_motion

def geometric_brownian_motion(
    T=1.0,
    N=1000,
    M=1,
    *,
    S0=1.0,
    mu=0.05,
    sigma=0.2,
    seed=None
):
    """
    Simulate geometric Brownian motion (GBM) paths.

    The geometric Brownian motion satisfies the stochastic differential equation:

        dS_t = mu * S_t dt + sigma * S_t dW_t

    whose exact solution is:

        S_t = S0 * exp((mu - 0.5 * sigma^2) t + sigma W_t)

    where W_t is a standard Brownian motion.

    Parameters
    ----------
    T : float, default=1.0
        Time horizon.
    N : int, default=1000
        Number of time steps.
    M : int, default=1
        Number of simulated paths.
    S0 : float, default=1.0
        Initial asset price.
    mu : float, default=0.05
        Drift coefficient.
    sigma : float, default=0.2
        Volatility coefficient.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    t : ndarray of shape (N+1,)
        Time grid.
    S : ndarray
        Simulated GBM paths.
        Shape is (N+1,) if M=1, otherwise (M, N+1).

    Notes
    -----
    This implementation uses the exact solution of the GBM SDE and relies on
    a standard Brownian motion generator.
    """
    
    t, W = brownian_motion(T, N, M, seed=seed)
    drift = (mu - 0.5 * sigma**2) * t

    S = S0 * np.exp(drift + sigma * W)

    return t, S