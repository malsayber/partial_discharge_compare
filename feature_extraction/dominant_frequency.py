"""Dominant frequency feature for partial discharge signals."""
from __future__ import annotations

import numpy as np


def dominant_frequency(signal: np.ndarray, fs: float = 1.0) -> float:
    """Return the dominant frequency component using FFT."""
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal))
    return float(freqs[np.argmax(spectrum)])
