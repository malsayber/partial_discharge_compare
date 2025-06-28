"""Base module for partial discharge feature extraction."""
from __future__ import annotations

import numpy as np


def extract_feature(signal: np.ndarray) -> float:
    """Example feature extraction placeholder.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    float
        Placeholder value representing extracted feature.
    """
    return float(signal.mean())

