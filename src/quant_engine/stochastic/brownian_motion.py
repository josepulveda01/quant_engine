import numpy as np

def brownian_motion(
    T=1.0,
    N=1000,
    M=1,
    mu=0.0,
    sigma=1.0,
    seed=None
):
    
    """
    Simulate standard Brownian motion (SBM) paths.

    W(0) = 0
    dW ~ N(0, dt)

    Parameters
    ----------
    T : float
        Time horizon
    N : int
        Number of time steps
    M : int
        Number of paths
    seed : int | None
        Random seed

    Returns
    -------
    t : ndarray
        Time grid
    W : ndarray
        Brownian paths
    """

    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    if M == 1:
        # Fast 1D generation
        dW = rng.normal(mu * dt, sigma * np.sqrt(dt), size=N)
        W = np.empty(N + 1)
        W[0] = 0.0
        W[1:] = np.cumsum(dW)
    else:
        # Vectorized multi-path generation
        dW = rng.normal(mu * dt, sigma * np.sqrt(dt), size=(M, N))
        W = np.empty((M, N + 1))
        W[:, 0] = 0.0
        W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W