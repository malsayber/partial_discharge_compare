"""Root mean square feature for partial discharge signals."""
from __future__ import annotations

import numpy as np


def time_rms(signal: np.ndarray) -> float:
    """Return the root mean square of ``signal``."""
    return float(np.sqrt(np.mean(signal ** 2)))
