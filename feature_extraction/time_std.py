"""Standard deviation feature for partial discharge signals."""
from __future__ import annotations

import numpy as np


def time_std(signal: np.ndarray) -> float:
    """Return the standard deviation of ``signal``."""
    return float(np.std(signal))
