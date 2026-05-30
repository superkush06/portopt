"""Covariance estimators."""

from __future__ import annotations

import numpy as np


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Standard sample covariance, columns are assets."""
    return np.cov(returns, rowvar=False, ddof=1)


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of sample covariance toward scaled identity.

    Returns (shrunk_cov, shrinkage_intensity in [0,1]).

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices."
    """
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    if T < 2 or N < 1:
        raise ValueError("need at least 2 rows and 1 column")

    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / T          # ML estimator (biased) for the math
    # Target: scaled identity with same mean variance.
    mu = np.trace(sample) / N
    F = mu * np.eye(N)

    # Frobenius-norm decomposition
    Y = Xc * Xc
    phi_mat = (Y.T @ Y) / T - sample * sample
    phi = phi_mat.sum()
    gamma = np.linalg.norm(sample - F, ord="fro") ** 2

    kappa = phi / gamma if gamma > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))
    shrunk = delta * F + (1.0 - delta) * sample
    return shrunk, float(delta)
