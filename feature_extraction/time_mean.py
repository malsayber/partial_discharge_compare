"""Mean value feature for partial discharge signals."""
from __future__ import annotations

import numpy as np


def time_mean(signal: np.ndarray) -> float:
    """Return the mean of ``signal``.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    float
        Mean value of the signal.
    """
    return float(np.mean(signal))
