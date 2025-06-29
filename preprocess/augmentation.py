from __future__ import annotations

"""Data augmentation helpers for time-series signals."""

import numpy as np
from tsaug import TimeWarp, AddNoise


def time_warp(x: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Apply random time warping using :mod:`tsaug`.

    Parameters
    ----------
    x:
        Input signal.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Augmented signal.
    """
    aug = TimeWarp(n_speed_change=2, max_speed_ratio=3, seed=seed)
    return aug.augment(x)


def add_jitter(x: np.ndarray, sigma: float = 0.01, seed: int | None = None) -> np.ndarray:
    """Add Gaussian noise (jitter) to a signal."""
    aug = AddNoise(scale=sigma, seed=seed)
    return aug.augment(x)
