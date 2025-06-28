"""Spectral entropy feature for partial discharge signals."""
from __future__ import annotations

import numpy as np
from scipy.stats import entropy


def spectral_entropy(signal: np.ndarray, fs: float = 1.0) -> float:
    """Return spectral entropy computed from the power spectrum."""
    power = np.abs(np.fft.rfft(signal)) ** 2
    psd = power / np.sum(power)
    return float(entropy(psd))
