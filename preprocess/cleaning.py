from __future__ import annotations

"""Signal cleaning utilities."""

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Apply a Butterworth band-pass filter."""
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def advanced_denoise(x: np.ndarray, method: str = "vmd") -> np.ndarray:
    """Placeholder for advanced denoising."""
    return x
